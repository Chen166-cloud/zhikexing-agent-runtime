"""版本化入库和带工作空间过滤的混合检索。"""

import asyncio
import hashlib
import io
import math
import re
import tempfile
import uuid
from pathlib import Path

from pgvector.sqlalchemy import Vector
from pypdf import PdfReader
from sqlalchemy import cast, delete, select

from .config import Settings
from .db import Chunk, Database, Document, DocumentVersion, new_id
from .provider import ModelProvider, lexical_terms
from .storage import ObjectStorage


def parse_pages(content: bytes, filename: str, parser: str) -> list[tuple[int, str]]:
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md"}:
        return [(1, content.decode("utf-8-sig"))]
    if suffix != ".pdf":
        raise ValueError("目前支持 PDF、UTF-8 TXT 和 Markdown 文件")
    if parser == "docling":
        from docling.document_converter import DocumentConverter

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "document.pdf"
            path.write_bytes(content)
            result = DocumentConverter().convert(path)
            pages = {}
            for item, _ in result.document.iterate_items():
                page = item.prov[0].page_no if getattr(item, "prov", []) else 1
                value = getattr(item, "text", "")
                if hasattr(item, "export_to_markdown"):
                    value = item.export_to_markdown(doc=result.document)
                if value:
                    pages.setdefault(page, []).append(value)
            return [(page, "\n".join(values)) for page, values in sorted(pages.items())]
    reader = PdfReader(io.BytesIO(content))
    if len(reader.pages) > 300:
        raise ValueError("单文档最多 300 页，请拆分后上传")
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def split_chunks(pages: list[tuple[int, str]]) -> list[dict]:
    """按页和中文句边界切块，保留 100 字符重叠用于跨句检索。"""
    chunks = []
    for page, content in pages:
        content = re.sub(r"[ \t]+", " ", content).strip()
        start = 0
        while start < len(content):
            end = min(start + 800, len(content))
            if end < len(content):
                boundaries = [content.rfind(c, start + 400, end) for c in "。！？\n"]
                boundary = max(boundaries)
                if boundary > start:
                    end = boundary + 1
            chunk = content[start:end].strip()
            if chunk:
                chunks.append({"page": page, "content": chunk, "ordinal": len(chunks)})
            if end >= len(content):
                break
            start = end - 100
    if not chunks:
        raise ValueError("未提取到文本；扫描件请启用 Docling/OCR 解析器后重试")
    if len(chunks) > 1500:
        raise ValueError("切块数量超过单文档限额，请拆分文件")
    return chunks


class KnowledgeService:
    def __init__(
        self, db: Database, provider: ModelProvider, storage: ObjectStorage, settings: Settings
    ):
        self.db, self.provider, self.storage, self.settings = db, provider, storage, settings
        self.reranker = None

    async def cleanup(self, document_id: str):
        """先撤销可见性，再清理对象；失败会由持久化调度下一轮重试。"""
        async with self.db.session() as session:
            versions = (
                await session.scalars(
                    select(DocumentVersion).where(DocumentVersion.document_id == document_id)
                )
            ).all()
        for version in versions:
            await self.storage.delete(version.object_key)
        async with self.db.session() as session:
            document = await session.get(Document, document_id)
            document.status = "DELETED"
            await session.commit()

    async def ingest(self, document_id: str):
        async with self.db.session() as session:
            document = await session.get(Document, document_id)
            if not document or document.deleted:
                return
            version = (
                await session.scalars(
                    select(DocumentVersion).where(
                        DocumentVersion.document_id == document_id,
                        DocumentVersion.version == document.pending_version,
                    )
                )
            ).one()
            version_id, number, workspace_id, title, object_key = (
                version.id,
                version.version,
                document.workspace_id,
                document.title,
                version.object_key,
            )
            version.attempt += 1
            version.status = document.status = "PARSING"
            document.error = None
            await session.commit()
        try:
            content = await self.storage.get(object_key)
            pages = await asyncio.to_thread(parse_pages, content, title, self.settings.parser)
            parts = await asyncio.to_thread(split_chunks, pages)
            async with self.db.session() as session:
                doc = await session.get(Document, document_id)
                if doc.deleted:
                    return
                doc.status = "EMBEDDING"
                await session.commit()
            vectors, used = [], 0
            for offset in range(0, len(parts), 10):
                batch, tokens = await self.provider.embed(
                    [x["content"] for x in parts[offset : offset + 10]]
                )
                vectors.extend(batch)
                used += tokens or 0
            # 完整新版本在一个事务里发布；失败期间旧 activeVersion 仍可检索。
            async with self.db.session() as session:
                doc = await session.get(Document, document_id, with_for_update=True)
                if doc.deleted or doc.pending_version != number:
                    return
                await session.execute(
                    delete(Chunk).where(Chunk.document_id == document_id, Chunk.version == number)
                )
                for part, vector in zip(parts, vectors):
                    identifier = str(
                        uuid.uuid5(uuid.NAMESPACE_URL, f"{version_id}/{part['ordinal']}")
                    )
                    session.add(
                        Chunk(
                            id=identifier,
                            document_id=document_id,
                            workspace_id=workspace_id,
                            version=number,
                            embedding=vector,
                            terms=lexical_terms(part["content"]),
                            **part,
                        )
                    )
                version = await session.get(DocumentVersion, version_id)
                version.status, version.embedding_tokens = "READY", used
                version.parser_version = self.settings.parser + "-structural-v1"
                doc.active_version, doc.status, doc.chunk_count, doc.error = (
                    number,
                    "READY",
                    len(parts),
                    None,
                )
                await session.commit()
        except Exception as exc:
            async with self.db.session() as session:
                doc = await session.get(Document, document_id)
                version = await session.get(DocumentVersion, version_id)
                if not doc.deleted:
                    doc.status, doc.error = "FAILED", (
                        str(exc)[:500]
                        if isinstance(exc, ValueError)
                        else "文档处理失败，请检查解析器、模型连接或存储后重试"
                    )
                    version.status = "FAILED"
                    await session.commit()

    async def retrieve(
        self, workspace_id: str, knowledge_base_ids: list[str], query: str, limit: int = 5
    ) -> list[dict]:
        if not knowledge_base_ids:
            return []
        query_vectors, _ = await self.provider.embed([query])
        vector = query_vectors[0]
        clauses = (
            Document.workspace_id == workspace_id,
            Document.knowledge_base_id.in_(knowledge_base_ids),
            Document.deleted.is_(False),
            Document.active_version > 0,
            Chunk.version == Document.active_version,
            Chunk.workspace_id == workspace_id,
        )
        async with self.db.session() as session:
            base = (
                select(Chunk, Document.title)
                .join(Document, Chunk.document_id == Document.id)
                .where(*clauses)
            )
            if self.db.postgres:
                distance = cast(Chunk.embedding, Vector(1024)).cosine_distance(vector)
                dense_rows = (await session.execute(base.order_by(distance).limit(20))).all()
            else:
                # SQLite 仅作为本地验证后端；正式 PostgreSQL 直接执行向量排序。
                candidates = (await session.execute(base)).all()
                dense_rows = sorted(
                    candidates, key=lambda row: self.cosine(vector, row[0].embedding), reverse=True
                )[:20]
            lexical_rows = (await session.execute(base)).all()
        terms = set(lexical_terms(query))
        ranked = sorted(
            lexical_rows, key=lambda row: len(terms.intersection(row[0].terms)), reverse=True
        )
        lexical = [row for row in ranked if terms.intersection(row[0].terms)][:20]
        scores, rows = {}, {}
        for ranking in (dense_rows, lexical):
            for rank, row in enumerate(ranking, 1):
                chunk = row[0]
                scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (60 + rank)
                rows[chunk.id] = row
        selected = [rows[key] for key in sorted(scores, key=scores.get, reverse=True)[:30]]
        if self.settings.rerank and selected:

            def rerank():
                if self.reranker is None:
                    from FlagEmbedding import FlagReranker

                    self.reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)
                result = self.reranker.compute_score([[query, row[0].content] for row in selected])
                return sorted(zip(result, selected), key=lambda pair: pair[0], reverse=True)

            selected = [row for _, row in await asyncio.to_thread(rerank)]
        return [
            {
                "evidenceId": f"E{i+1}",
                "chunkId": row[0].id,
                "documentId": row[0].document_id,
                "title": row[1],
                "page": row[0].page,
                "content": row[0].content,
                "version": row[0].version,
                "score": scores[row[0].id],
            }
            for i, row in enumerate(selected[:limit])
        ]

    async def authorized_evidence(self, workspace_id: str, evidence: list[dict]) -> list[dict]:
        if not evidence:
            return []
        async with self.db.session() as session:
            current = (
                await session.scalars(
                    select(Document).where(
                        Document.workspace_id == workspace_id,
                        Document.deleted.is_(False),
                        Document.id.in_([e["documentId"] for e in evidence]),
                    )
                )
            ).all()
            versions = {doc.id: doc.active_version for doc in current}
        return [e for e in evidence if versions.get(e["documentId"]) == e["version"]]

    @staticmethod
    def cosine(left, right):
        denominator = math.sqrt(sum(x * x for x in left) * sum(x * x for x in right))
        return sum(x * y for x, y in zip(left, right)) / denominator if denominator else 0
