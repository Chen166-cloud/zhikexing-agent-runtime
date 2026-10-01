"""Load isolated fixture identity and, when requested, existing model credentials.

Run inside the benchmark-only container. Never print credentials or read business
database settings from the existing .env. Each workload remains explicitly gated.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path


def configure(bootstrap: Path, real: bool):
    data = json.loads(bootstrap.read_text(encoding="utf-8-sig"))
    owner = data["actors"][0]
    os.environ.update({"AGENT_EXPERIMENT_AUTH_TOKEN": data["ownerToken"],
                       "AGENT_EXPERIMENT_ACTOR_ID": owner["actorId"],
                       "AGENT_EXPERIMENT_WORKSPACE_ID": data["workspaceId"],
                       "AGENT_TOKEN_LIMIT": "40000", "AGENT_COST_LIMIT_CNY": "0.10"})
    version = bootstrap.parent / "code-version.json"
    if version.exists():
        info = json.loads(version.read_text(encoding="utf-8"))
        os.environ["AGENT_EXPERIMENT_REPOSITORY_COMMIT"] = info["agentCommit"]
        os.environ["AGENT_EXPERIMENT_CODE_SHA256"] = info["agentSourceHash"]
    if real:
        allowed = {"DASHSCOPE_API_KEY", "API-KEY", "DASHSCOPE_BASE_URL", "AI_CHAT_MODEL",
                   "AI_EMBEDDING_MODEL", "AI_EMBEDDING_DIMENSIONS"}
        source = Path(os.getenv("AGENT_EXPERIMENT_MODEL_ENV_FILE", "/run/benchmark-secrets/backend.env"))
        if source.is_file():
            for line in source.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                name, value = name.strip(), value.strip()
                if name in allowed and value:
                    if value[:1] in {"'", '"'} and value[-1:] == value[:1]:
                        value = value[1:-1]
                    os.environ.setdefault(name, value)
        if not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("API-KEY")):
            raise SystemExit("Existing model credential is unavailable; no external calls were made")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=Path, default=Path("experiments/results/2026-10-01/bootstrap.json"))
    parser.add_argument("stage", choices=["recovery", "evaluate", "rag"])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    real = "bailian" in args.arguments
    configure(args.bootstrap, real)
    sys.argv = ["experiments." + args.stage] + args.arguments
    runpy.run_module("experiments." + args.stage, run_name="__main__")


if __name__ == "__main__":
    main()
