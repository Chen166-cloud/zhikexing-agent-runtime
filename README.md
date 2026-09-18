# Intelligent Agent Runtime

企业知识与课程预约平台的独立 Python 服务，与 Java 业务后端、Vue 工作台通过 HTTP 联动。默认使用百炼 `qwen3.7-flash` 与 `text-embedding-v4`（1024 维），共用一把百炼 API Key。

## 三个项目的位置

| 项目 | 本机目录 | 职责 |
|---|---|---|
| Java 后端 | `D:/java/SpringAI/intelligent-integrated-interaction-platform` | 登录、工作空间成员、业务工具、审批、幂等预约、可靠投递、对外 API |
| Python 运行时 | `D:/develop/intelligent-agent-runtime` | LangGraph、持久化恢复、完整消息、知识库入库/检索、模型调用、评测 |
| Vue 前端 | `D:/develop/web-intelligent-integrated-interaction-platform` | 运行轨迹、审批卡、引用预览、知识管理、评测和费用展示 |

## 启动

Python 3.13 为当前验证版本。建议通过 Java 仓库的 Docker Compose 启动整套应用和基础设施，详见该仓库 `docs/deployment/`；它使用本项目作为独立构建上下文。

本机开发安装：

```powershell
cd D:\develop\intelligent-agent-runtime
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
# 配置下表变量后启动；直接窗口运行，便于开发时停止。
.\.venv\Scripts\python.exe -m iiip_agent
```

| 变量 | 说明 |
|---|---|
| `DASHSCOPE_API_KEY` | 百炼模型 Key；兼容本机已有 `API-KEY`，不需要第二家模型凭据 |
| `AGENT_INTERNAL_TOKEN` | Java/Python 共用的内部服务凭据，必须配置；与模型 Key 不同 |
| `AGENT_DATABASE_URL` | 正式环境 `postgresql+psycopg://用户:密码@主机:端口/数据库`；缺省 SQLite 仅用于本地验证 |
| `JAVA_TOOL_BASE_URL` | Java 地址，本机 Compose 映射为 `http://localhost:18080`，容器内为 `http://backend:8080` |
| `AI_CHAT_MODEL` / `AI_EMBEDDING_MODEL` | 默认 `qwen3.7-flash` / `text-embedding-v4` |
| `AI_EMBEDDING_DIMENSIONS` | 固定 1024，变更需迁移索引 |
| `PORT` / `STORAGE_ROOT` | 默认 8000 / `./data`；数据目录需持久化 |
| `AGENT_COST_LIMIT_CNY` | 默认 0.10 元；每次模型请求前检查已知估算费用，单次请求仍可能越过阈值，不能当作厂商硬计费上限 |
| `S3_ENDPOINT_URL` / `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY` / `S3_BUCKET` | 可选 S3 对象存储；未设置 endpoint 时存本地数据目录；默认 bucket 为 `iiip-documents` |
| `RABBITMQ_ENABLED` / `RABBITMQ_URL` | 开启可靠队列消费；关闭时 Java 使用明确配置的 HTTP outbox 投递 |
| `DOCUMENT_PARSER` | 默认 `pypdf`，支持有文本的 PDF/TXT/Markdown；安装 `.[documents]` 后可选 `docling` 做布局/OCR解析 |
| `AI_RERANK_ENABLED` | 默认 false；安装 `.[rerank]` 后可开启本地 BGE 对照，需要另测模型下载、CPU/内存与延迟 |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | 可选 OTLP HTTP Collector 地址 |
| `LANGFUSE_BASE_URL` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | 可选 Langfuse 导出，凭据仅用于观测，不是另一家模型 Key |

`AI_PROVIDER=fixture` 仅用于显式隔离验证，输出有 fixture 标记；正式演示使用默认 `bailian`。默认 `/health` 可查看实际 provider、模型和存储类型。

## 执行与恢复

LangGraph 运行 `reason → tools → reason` 的有界循环，预约草稿进入持久化 `interrupt`。用户批准后，运行时通过 Java 提交并查询结果；批准参数和事务幂等由 Java 掌握。重启后从同一 `runId` 的 checkpoint 恢复，已执行动作按 `actionId` 查询，不重复创建预约。

完整消息独立保存；模型只读取有预算的历史窗口。SSE 事件具有单调序号，支持 `after` 和 `Last-Event-ID`。终态、最终消息和结束事件在同一数据库事务提交。

当前图版本为 `react-approval-v3`。同一运行的相同只读查询缓存 30 秒，知识证据复用前再次检查授权和版本；不缓存预约草稿或写入。模型重复查询空结果时进入 `WAITING_INPUT`，用户补充后从 checkpoint 继续。引用必须使用实际检索到的证据编号；缺失或伪造编号只允许一次受预算约束的修正，前端通过 `message.reset` 清除旧草稿，仍无效则明确失败。

run 与 checkpoint 固定图版本、提示词和工具 Schema 哈希、模型与 Embedding 配置。恢复时不兼容会返回明确错误；先恢复原部署排空任务，再发布改变执行语义的新版本。费用与 Token 预算在新请求前检查，缺失厂商 usage 的费用仍显示未知。

当前使用**单 worker、多任务并发**；PostgreSQL advisory lock 阻止误开两个 worker 竞争 checkpoint。它不等同于已完成分布式多 worker fencing。生产部署采用 PostgreSQL/pgvector，SQLite 仅用于本地功能验证。

知识文档先异步解析/向量化，完整版本发布后才切换 activeVersion。查询先限定空间、知识库及有效版本，再融合 dense 与中文词法候选；词法基线按词项重叠排序，不能称为 Elasticsearch/BM25。删除先撤销读取和检索，再异步清理对象。

## 验证与可观测

- `/health`：数据库和运行配置；不调用计费模型。
- `/metrics`：Prometheus 运行/文档状态与后台作业数，不含用户内容。
- `/internal/v1/metrics`：授权空间的业务统计，未知账单明确标记，不当作免费调用。
- `/internal/v1/evaluations`：生产评测功能，保存逐例结果、规则判定、引用、延迟和 usage；没有参考答案的样本不自动算通过。

本次验证包括 SQLite 的 21 项功能回归、扩展后的真实 PostgreSQL/pgvector 37 项回归，以及百炼真实聊天、1024 维 Embedding 和知识评测冒烟。数据库回归使用显式模拟模型/业务工具，不能当作模型效果指标。37 项包含原有案例及新案例，不能与旧轮次叠加计算。临时测试类和脚本按用户要求在验证完成后清理，去敏结果与部署记录保留在 Java 项目文档中。

清理例外：自动审批拒绝删除 `.test-artifacts` 中的 4 个临时脚本和 1 个本地验证状态文件，仅返回 `blocked by policy`。文件仍留在该 Git/Docker 忽略目录，尚未清理完成；应用运行不依赖它们，详细登记见 Java 项目《开发与验证记录》。

同一数据库只允许一个 runtime worker。开发时若要使用 Compose 的数据库，应先停止 Compose 中的 `agent-runtime`；独立验证可使用另建数据库和存储目录，避免与运行中的任务竞争。
