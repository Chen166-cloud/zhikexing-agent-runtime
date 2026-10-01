"""Frozen-embedding RAG comparison. prepare is gated; compare never calls models.

Optional PostgreSQL mode uses a dedicated experiment schema, never production
knowledge tables. This small synthetic corpus is a correctness benchmark, not a
large-scale capacity measurement or a claim of real user retrieval accuracy.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import psycopg
from psycopg import sql

from zhikexing_agent.config import Settings
from zhikexing_agent.provider import ModelProvider, lexical_terms

from .common import append_json, dataset_hash, percentile, required, stamp, write_json


HERE = Path(__file__).resolve().parent
VARIANTS = ("dense", "lexical", "rrf")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(corpus, questions):
    chunks = {chunk["chunkId"]: chunk for chunk in corpus}
    if len(chunks) != len(corpus) or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate corpus/query ID")
    for question in questions:
        if not question["relevantChunkIds"] or set(question["relevantChunkIds"]) - chunks.keys():
            raise ValueError("Gold chunk labels missing or outside corpus")
        documents = {chunks[key]["documentId"] for key in question["relevantChunkIds"]}
        if documents != set(question["relevantDocumentIds"]):
            raise ValueError("Document and chunk gold labels disagree")


def cosine(left, right):
    denominator = math.sqrt(sum(x*x for x in left) * sum(x*x for x in right))
    return sum(x*y for x, y in zip(left, right)) / denominator if denominator else 0


def ranking_metrics(ranking, relevant, k=5):
    selected, relevant = ranking[:k], set(relevant)
    hits = relevant.intersection(selected)
    first = next((index for index, value in enumerate(selected, 1) if value in relevant), None)
    return {"recallAt5": len(hits) / len(relevant),
            "reciprocalRankAt5": 1 / first if first else 0,
            "allEvidenceAt5": hits == relevant}


def fuse(dense, lexical):
    scores = {}
    for ranking in (dense[:20], lexical[:20]):
        for rank, key in enumerate(ranking, 1):
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
    # Explicit secondary key prevents accidental dependence on input dict order.
    return sorted(scores, key=lambda key: (-scores[key], key))


async def prepare(args, corpus, questions):
    if args.cache.exists():
        raise ValueError("Frozen cache already exists; choose a new path")
    if args.provider == "bailian" and not args.allow_real_model:
        raise ValueError("Embedding API requires --allow-real-model")
    if args.max_texts < len(corpus) + len(questions):
        raise ValueError("Explicit --max-texts allowance is smaller than corpus + questions")
    settings = Settings(provider=args.provider, internal_token="experiment-offline-only",
                        storage_root=args.cache.parent, api_key=os.getenv("DASHSCOPE_API_KEY", os.getenv("API-KEY", "")))
    settings.validate()
    provider = ModelProvider(settings)
    cache = {"createdAt": stamp(), "provider": args.provider, "model": settings.embedding_model,
             "dimensions": settings.dimensions, "corpusSha256": dataset_hash(args.corpus),
             "questionsSha256": dataset_hash(args.questions), "chunks": {}, "queries": {},
             "runnerSha256": dataset_hash(Path(__file__)),
             "providerSourceSha256": dataset_hash(HERE.parent / "zhikexing_agent" / "provider.py"),
             "queryEmbeddingLatencyMode": "individual remote request per question; includes retries", "embeddingUsageTokens": 0,
             "embeddingCostCny": None}
    partial_path = args.cache.with_name(args.cache.name + ".partial.json")
    if partial_path.exists():
        if not args.resume_prepare:
            await provider.close()
            raise ValueError("Partial embedding cache exists; use --resume-prepare to avoid duplicate API charges")
        previous = json.loads(partial_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != cache.get(key) for key in ("provider", "model", "dimensions", "corpusSha256", "questionsSha256")):
            await provider.close()
            raise ValueError("Partial cache configuration does not match")
        cache = previous
    try:
        for offset in range(0, len(corpus), 10):
            batch = [chunk for chunk in corpus[offset:offset+10] if chunk["chunkId"] not in cache["chunks"]]
            if not batch:
                continue
            started = time.perf_counter()
            vectors, usage = await provider.embed([chunk["content"] for chunk in batch])
            latency = (time.perf_counter() - started) * 1000
            for chunk, vector in zip(batch, vectors):
                cache["chunks"][chunk["chunkId"]] = vector
            cache.setdefault("corpusBatches", []).append({"count": len(batch), "latencyMs": latency,
                                                         "usageTokens": usage})
            cache["embeddingUsageTokens"] = (cache["embeddingUsageTokens"] + usage
                                              if usage is not None and cache["embeddingUsageTokens"] is not None else None)
            write_json(partial_path, cache)
        for question in questions:
            if question["id"] in cache["queries"]:
                continue
            started = time.perf_counter()
            vectors, usage = await provider.embed([question["question"]])
            cache["queries"][question["id"]] = {"vector": vectors[0],
                "latencyMs": (time.perf_counter() - started) * 1000, "usageTokens": usage}
            cache["embeddingUsageTokens"] = (cache["embeddingUsageTokens"] + usage
                                              if usage is not None and cache["embeddingUsageTokens"] is not None else None)
            write_json(partial_path, cache)
        write_json(args.cache, cache)
    finally:
        await provider.close()


class MemoryRetriever:
    def __init__(self, corpus, vectors):
        self.corpus, self.vectors = corpus, vectors
        self.terms = {row["chunkId"]: set(lexical_terms(row["content"])) for row in corpus}

    def dense(self, vector):
        scores = [(cosine(vector, self.vectors[row["chunkId"]]), row["chunkId"]) for row in self.corpus]
        return [key for _, key in sorted(scores, key=lambda item: (-item[0], item[1]))[:20]]

    def lexical(self, question):
        query_terms = set(lexical_terms(question))
        scores = [(len(query_terms.intersection(self.terms[row["chunkId"]])), row["chunkId"]) for row in self.corpus]
        return [key for score, key in sorted(scores, key=lambda item: (-item[0], item[1]))[:20] if score > 0]

    def close(self):
        pass


class PostgresRetriever:
    def __init__(self, corpus, vectors, cache_hash, dimensions):
        if os.getenv("AGENT_EXPERIMENT_ISOLATED") != "yes":
            raise ValueError("PostgreSQL comparison requires AGENT_EXPERIMENT_ISOLATED=yes")
        dsn = required("AGENT_EXPERIMENT_DATABASE_URL").replace("postgresql+psycopg://", "postgresql://", 1)
        self.connection = psycopg.connect(dsn, autocommit=True)
        self.schema = "experiment_rag_" + cache_hash[:12]
        self.table = sql.Identifier(self.schema, "chunks")
        with self.connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self.schema)))
            cursor.execute(sql.SQL("CREATE TABLE IF NOT EXISTS {} (chunk_id text PRIMARY KEY, document_id text NOT NULL, "
                                   "content text NOT NULL, terms jsonb NOT NULL, embedding vector({}))").format(
                                       self.table, sql.Literal(dimensions)))
            for row in corpus:
                cursor.execute(sql.SQL("INSERT INTO {} VALUES(%s,%s,%s,%s::jsonb,%s::vector) ON CONFLICT(chunk_id) DO NOTHING").format(self.table),
                               (row["chunkId"], row["documentId"], row["content"],
                                json.dumps(lexical_terms(row["content"]), ensure_ascii=False), json.dumps(vectors[row["chunkId"]])))
            cursor.execute(sql.SQL("ANALYZE {}").format(self.table))

    def dense(self, vector):
        with self.connection.cursor() as cursor:
            cursor.execute(sql.SQL("SELECT chunk_id FROM {} ORDER BY embedding <=> %s::vector,chunk_id LIMIT 20").format(self.table),
                           (json.dumps(vector),))
            return [row[0] for row in cursor.fetchall()]

    def lexical(self, question):
        # Deliberately mirror current production full-candidate transfer + Python overlap.
        with self.connection.cursor() as cursor:
            cursor.execute(sql.SQL("SELECT chunk_id,terms FROM {}").format(self.table))
            candidates = cursor.fetchall()
        terms = set(lexical_terms(question))
        scores = [(len(terms.intersection(row[1])), row[0]) for row in candidates]
        return [key for score, key in sorted(scores, key=lambda item: (-item[0], item[1]))[:20] if score > 0]

    def close(self):
        self.connection.close()


def compare(args, corpus, questions):
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    if cache["corpusSha256"] != dataset_hash(args.corpus) or cache["questionsSha256"] != dataset_hash(args.questions):
        raise ValueError("Frozen vector cache does not match the dataset")
    if set(cache["chunks"]) != {row["chunkId"] for row in corpus} or set(cache["queries"]) != {q["id"] for q in questions}:
        raise ValueError("Frozen cache missing entries")
    if any(len(vector) != cache["dimensions"] for vector in list(cache["chunks"].values()) + [row["vector"] for row in cache["queries"].values()]):
        raise ValueError("Embedding dimension mismatch")
    if (args.output / "cases.jsonl").exists():
        raise ValueError("Use a new output directory")
    retriever = (PostgresRetriever(corpus, cache["chunks"], dataset_hash(args.cache), cache["dimensions"])
                 if args.engine == "postgres" else MemoryRetriever(corpus, cache["chunks"]))
    selected = [q for q in questions if not args.split or q["split"] == args.split]
    # Warm library tokenizer/database paths once, outside the measured repetitions.
    if selected:
        retriever.dense(cache["queries"][selected[0]["id"]]["vector"])
        retriever.lexical(selected[0]["question"])
    rows = []
    try:
        for question in selected:
            vector = cache["queries"][question["id"]]["vector"]
            timings = {variant: [] for variant in VARIANTS}
            rankings = {}
            for repeat in range(args.repeats):
                variants = list(VARIANTS)
                random.Random(args.seed + repeat).shuffle(variants)
                for variant in variants:
                    started = time.perf_counter()
                    if variant == "dense":
                        ranking = retriever.dense(vector)
                    elif variant == "lexical":
                        ranking = retriever.lexical(question["question"])
                    else:
                        ranking = fuse(retriever.dense(vector), retriever.lexical(question["question"]))
                    timings[variant].append((time.perf_counter() - started) * 1000)
                    rankings[variant] = ranking
            for variant in VARIANTS:
                row = {"caseId": question["id"], "split": question["split"], "variant": variant,
                       "top5ChunkIds": rankings[variant][:5], "relevantChunkIds": question["relevantChunkIds"],
                       "relevantDocumentIds": question["relevantDocumentIds"],
                       **ranking_metrics(rankings[variant], question["relevantChunkIds"]),
                       "rankingLatencyMs": timings[variant],
                       "queryEmbeddingLatencyMs": cache["queries"][question["id"]]["latencyMs"]}
                rows.append(row)
                append_json(args.output / "cases.jsonl", row)
        summaries = {}
        for split in ("development", "holdout", "all"):
            summaries[split] = {}
            for variant in VARIANTS:
                group = [row for row in rows if row["variant"] == variant and (split == "all" or row["split"] == split)]
                if not group:
                    continue
                summaries[split][variant] = {"queries": len(group),
                    "macroRecallAt5": sum(row["recallAt5"] for row in group) / len(group),
                    "MRRAt5": sum(row["reciprocalRankAt5"] for row in group) / len(group),
                    "allEvidenceAt5Rate": sum(row["allEvidenceAt5"] for row in group) / len(group),
                    "retrievalP95Ms": percentile([value for row in group for value in row["rankingLatencyMs"]]),
                    "queryEmbeddingP95Ms": percentile([row["queryEmbeddingLatencyMs"] for row in group])}
        write_json(args.output / "summary.json", {"provider": cache["provider"], "engine": args.engine,
                   "corpusChunks": len(corpus), "questions": len(selected), "model": cache["model"],
                   "dimensions": cache["dimensions"], "cacheSha256": dataset_hash(args.cache),
                   "embeddingUsageTokens": cache["embeddingUsageTokens"], "embeddingCostCny": None,
                   "externalModelCallsDuringCompare": 0, "repeats": args.repeats,
                   "latencyScope": "PostgreSQL exact vector query / all candidate transfer + Python lexical / both for RRF; excludes embedding, generation, HTTP and authorization" if args.engine == "postgres" else "in-memory ranking only; excludes PostgreSQL, embedding, HTTP and generation",
                   "semanticEvidence": cache["provider"] != "fixture", "metrics": summaries,
                   "limitations": ["synthetic initial correctness dataset", "no ANN index or reranker", "fixture hash vectors do not measure semantic model quality", "not production full retrieval latency"]})
    finally:
        retriever.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["validate", "prepare", "compare"])
    parser.add_argument("--corpus", type=Path, default=HERE / "rag_corpus.jsonl")
    parser.add_argument("--questions", type=Path, default=HERE / "rag_questions.jsonl")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--provider", choices=["fixture", "bailian"], default="fixture")
    parser.add_argument("--allow-real-model", action="store_true")
    parser.add_argument("--resume-prepare", action="store_true")
    parser.add_argument("--max-texts", type=int, default=150)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--engine", choices=["memory", "postgres"], default="memory")
    parser.add_argument("--split", choices=["development", "holdout"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    corpus, questions = read_jsonl(args.corpus), read_jsonl(args.questions)
    validate(corpus, questions)
    if args.mode == "validate":
        print(json.dumps({"chunks": len(corpus), "questions": len(questions),
                          "development": sum(q["split"] == "development" for q in questions),
                          "holdout": sum(q["split"] == "holdout" for q in questions)}, ensure_ascii=False))
        return 0
    if args.cache is None or (args.mode == "compare" and args.output is None):
        parser.error("--cache required; compare also requires --output")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.mode == "prepare":
        asyncio.run(prepare(args, corpus, questions))
    else:
        compare(args, corpus, questions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
