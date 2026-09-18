"""环境配置集中读取；模型凭据不出现在接口和日志中。"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "AGENT_DATABASE_URL", os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/agent.db")
        )
    )
    internal_token: str = field(default_factory=lambda: os.getenv("AGENT_INTERNAL_TOKEN", ""))
    api_key: str = field(
        default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", os.getenv("API-KEY", ""))
    )
    api_base: str = field(
        default_factory=lambda: os.getenv(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).rstrip("/")
    )
    chat_model: str = field(default_factory=lambda: os.getenv("AI_CHAT_MODEL", "qwen3.7-flash"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("AI_EMBEDDING_MODEL", "text-embedding-v4")
    )
    dimensions: int = field(
        default_factory=lambda: int(os.getenv("AI_EMBEDDING_DIMENSIONS", "1024"))
    )
    provider: str = field(default_factory=lambda: os.getenv("AI_PROVIDER", "bailian"))
    java_url: str = field(
        default_factory=lambda: os.getenv("JAVA_TOOL_BASE_URL", "http://localhost:8080").rstrip("/")
    )
    storage_root: Path = field(default_factory=lambda: Path(os.getenv("STORAGE_ROOT", "./data")))
    max_steps: int = 12
    max_tools: int = 8
    max_tokens: int = field(default_factory=lambda: int(os.getenv("AGENT_TOKEN_LIMIT", "40000")))
    max_cost_cny: float = field(
        default_factory=lambda: float(os.getenv("AGENT_COST_LIMIT_CNY", "0.10"))
    )
    run_timeout: int = field(
        default_factory=lambda: int(os.getenv("AGENT_RUN_TIMEOUT_SECONDS", "240"))
    )
    max_parallel: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_PARALLEL", "4")))
    s3_endpoint: str = field(default_factory=lambda: os.getenv("S3_ENDPOINT_URL", ""))
    s3_bucket: str = field(default_factory=lambda: os.getenv("S3_BUCKET", "iiip-documents"))
    parser: str = field(default_factory=lambda: os.getenv("DOCUMENT_PARSER", "pypdf"))
    rerank: bool = field(
        default_factory=lambda: os.getenv("AI_RERANK_ENABLED", "false").lower() == "true"
    )

    def validate(self) -> None:
        if not self.internal_token:
            raise ValueError("请配置 AGENT_INTERNAL_TOKEN")
        if self.provider not in {"bailian", "fixture"}:
            raise ValueError("AI_PROVIDER 只支持 bailian 或显式测试模式 fixture")
        if self.provider == "bailian" and not self.api_key:
            raise ValueError("请配置 DASHSCOPE_API_KEY（兼容已有 API-KEY）")
        if self.dimensions != 1024:
            raise ValueError("当前索引固定为 1024 维；变更维度需要迁移索引")
        self.storage_root.mkdir(parents=True, exist_ok=True)
