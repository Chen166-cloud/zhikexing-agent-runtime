# Intelligent Agent Runtime

企业知识、课程预约与免费试听平台的独立 Python 服务，与 Java 业务后端、Vue 工作台通过 HTTP 联动。默认使用百炼 `qwen3.7-flash` 与 `text-embedding-v4`（1024 维），共用一把百炼 API Key。

截至 2026-09-25，当前 Agent 图为 `react-trial-approval-v4`。普通命令由 Java 的 RocketMQ 消费者经内部 HTTP 幂等交给本服务，试听草稿须经用户批准，再查询 Java 参与请求的最终订单结果；本服务不持有业务库存。新版已在隔离环境完成跨服务 fixture 验证，常用完整 Compose 应用尚未重建验收。

三端功能、业务状态、权限和实际接口边界见 [产品功能说明书（研发版）](https://gitee.com/chy66666/intelligent-integrated-interaction-platform/blob/master/docs/product/产品功能说明书-研发版.md)。

## 三个独立项目

| 项目 | Git 仓库 | 职责 |
|---|---|---|
| Java 后端 | [intelligent-integrated-interaction-platform](https://gitee.com/chy66666/intelligent-integrated-interaction-platform.git) | 登录、工作空间成员、对外 API、审批、业务幂等与 outbox |
| Python 运行时 | [intelligent-agent-runtime](https://gitee.com/chy66666/intelligent-agent-runtime.git) | LangGraph、checkpoint、事件、知识库、模型适配与评测 |
| Vue 前端 | [web-intelligent-integrated-interaction-platform](https://gitee.com/chy66666/web-intelligent-integrated-interaction-platform.git) | 工作台、SSE、审批卡、引用预览、知识与评测管理 |

推荐在任意开发目录下将三个仓库克隆为同级目录：

```sh
git clone https://gitee.com/chy66666/intelligent-integrated-interaction-platform.git
git clone https://gitee.com/chy66666/intelligent-agent-runtime.git
git clone https://gitee.com/chy66666/web-intelligent-integrated-interaction-platform.git
```

Java 仓库的 Compose 默认从同级目录获取 Python、Vue 构建上下文；非同级存放时，在 Java 仓库的本地 `.env` 配置 `AGENT_RUNTIME_PATH` 和 `FRONTEND_PATH`，无需修改代码。三个服务通过可配置的 HTTP 地址联动，不依赖开发者电脑上的固定路径。

## 启动

Python 3.13 为当前验证版本。建议通过 Java 仓库的 Docker Compose 启动整套应用和基础设施，详见该仓库 `docs/deployment/`；它使用本项目作为独立构建上下文。

本机开发安装（在克隆得到的 `intelligent-agent-runtime` 仓库根目录执行）：

```powershell
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
| `DOCUMENT_PARSER` | 默认 `pypdf`，支持有文本的 PDF/TXT/Markdown；安装 `.[documents]` 后可选 `docling` 做布局/OCR解析 |
| `AI_RERANK_ENABLED` | 默认 false；安装 `.[rerank]` 后可开启本地 BGE 对照，需要另测模型下载、CPU/内存与延迟 |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | 可选 OTLP HTTP Collector 地址 |
| `LANGFUSE_BASE_URL` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | 可选 Langfuse 导出，凭据仅用于观测，不是另一家模型 Key |

`AI_PROVIDER=fixture` 仅用于显式隔离验证，输出有 fixture 标记；正式演示使用默认 `bailian`。默认 `/health` 可查看实际 provider、模型和存储类型。

## RocketMQ 命令投递边界

统一消息链路由 Java 管理：业务 outbox → RocketMQ → Java 命令消费者 → Python `/internal/v1/runs` 或 `/internal/v1/runs/{id}/cancel`。消费者只有在 Python 返回持久化成功后才确认消息，网络异常由 Java 的消费重试处理。Python 不再直接消费 broker，也不需要 MQ SDK、连接凭据或独立队列开关。

内部 HTTP 保持 `X-Internal-Token`、`X-Actor-Id`、`X-Workspace-Id` 和 `X-Command-Delivery: true` 契约。Python 以 runId 主键和 request_hash 对创建命令去重，在同一事务保存任务和输入；取消状态及终态事件同样在返回前提交。这里没有独立的 command inbox 表，不把 HTTP 返回前持久化等同于端到端 exactly-once。不要绕过 Java 的可信身份校验向运行时直接开放公网请求。

## 执行与恢复

LangGraph 运行 `reason → tools → reason` 的有界循环，预约草稿进入持久化 `interrupt`。用户批准后，运行时通过 Java 提交并查询结果；批准参数和事务幂等由 Java 掌握。重启后从同一 `runId` 的 checkpoint 恢复，已执行动作按 `actionId` 查询，不重复创建预约。

完整消息独立保存；模型只读取有预算的历史窗口。SSE 事件具有单调序号，支持 `after` 和 `Last-Event-ID`。终态、最终消息和结束事件在同一数据库事务提交。

当前图版本为 `react-trial-approval-v4`。同一运行的相同课程/校区/知识查询缓存 30 秒，知识证据复用前再次检查授权和版本；不缓存预约草稿、写入、试听活动与申请状态。模型重复查询空结果时进入 `WAITING_INPUT`，用户补充后从 checkpoint 继续。引用必须使用实际检索到的证据编号；缺失或伪造编号只允许一次受预算约束的修正，前端通过 `message.reset` 清除旧草稿，仍无效则明确失败。

run 与 checkpoint 固定图版本、提示词和工具 Schema 哈希、模型与 Embedding 配置。恢复时不兼容会返回明确错误；先恢复原部署排空任务，再发布改变执行语义的新版本。费用与 Token 预算在新请求前检查，缺失厂商 usage 的费用仍显示未知。

### 免费试听工具与审批

| 模型可用工具 | Java 内部接口 | 行为 |
|---|---|---|
| `query_trial_campaigns` | `GET /internal/v1/tools/trial-campaigns` | 查询活动；不缓存即时窗口/余量 |
| `draft_trial_claim` | `POST /internal/v1/tools/draft-trial-claim` | 只接受 campaignId，运行时生成稳定 actionId；不提交、不预占 |
| `query_trial_claim` | `GET /internal/v1/tools/trial-claims/by-action/{actionId}` | 查询当前用户申请的真实结果，不重新申请 |

试听草稿返回 `toolName=claim_trial` 的审批对象，图在持久化 interrupt 处等待用户批准。批准后才调用 `POST /internal/v1/tools/execute-trial-claim`，只提交原 actionId 和 approvalId；用户、空间与 run 由运行时可信请求头携带。旧普通预约缺省 `toolName=reserve_course`，继续使用原预约接口。

执行接口受理之后，图按 actionId 最多查询 3 次（间隔 1 秒），不会因排队或响应慢重新申请。`PENDING`/`RESERVED` 明确显示仍在处理中；只有查询返回 `SUCCEEDED` 并带真实 orderId 才显示成功；`REJECTED` 显示失败原因。回执由确定性节点生成，不交给模型改写业务状态。用户可在后续任务里提供动作编号继续查询，Agent 完成一次查询不代表试听订单已经成功。

返回字段约定为 actionId、requestId、campaignId、status、可空 orderId/reason；运行时另附 confirmed 与状态说明。停止 Agent 只停止运行和后续等待，**不会撤销已受理的试听申请**。试听费用为 0，本模块不提供订单取消、付款或退款流程。开场前可以准备草稿，是否在有效窗口及有名额由 Java 提交时校验。

当前使用**单 worker、多任务并发**；PostgreSQL advisory lock 阻止误开两个 worker 竞争 checkpoint。它不等同于已完成分布式多 worker fencing。生产部署采用 PostgreSQL/pgvector，SQLite 仅用于本地功能验证。

知识文档先异步解析/向量化，完整版本发布后才切换 activeVersion。查询先限定空间、知识库及有效版本，再融合 dense 与中文词法候选；词法基线按词项重叠排序，不能称为 Elasticsearch/BM25。删除先撤销读取和检索，再异步清理对象。

## 验证与可观测

持续维护的隔离回归使用临时 SQLite、fixture 模型和模拟 Java HTTP，不访问真实模型、MQ、业务库或本地 `.env`：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

覆盖命令重投/冲突/取消、可信身份、试听批准与拒绝、排队状态、售罄、checkpoint 重启后对账、稳定 actionId、状态不缓存、旧普通预约兼容。真实 RocketMQ 与 Java 消费重试的联调在 Java 仓库执行，不能把这些隔离测试写成 broker 集成验证。

- `/health`：数据库和运行配置；不调用计费模型。
- `/metrics`：Prometheus 运行/文档状态与后台作业数，不含用户内容。
- `/internal/v1/metrics`：授权空间的业务统计，未知账单明确标记，不当作免费调用。
- `/internal/v1/evaluations`：生产评测功能，保存逐例结果、规则判定、引用、延迟和 usage；没有参考答案的样本不自动算通过。

本次验证包括 SQLite 的 21 项功能回归、扩展后的真实 PostgreSQL/pgvector 37 项回归，以及百炼真实聊天、1024 维 Embedding 和知识评测冒烟。数据库回归使用显式模拟模型/业务工具，不能当作模型效果指标。37 项包含原有案例及新案例，不能与旧轮次叠加计算。临时测试类和脚本按用户要求在验证完成后清理，去敏结果与部署记录保留在 Java 项目文档中。

清理例外：自动审批拒绝删除 `.test-artifacts` 中的 4 个临时脚本和 1 个本地验证状态文件，仅返回 `blocked by policy`。文件仍留在该 Git/Docker 忽略目录，尚未清理完成；应用运行不依赖它们，详细登记见 Java 项目《开发与验证记录》。

同一数据库只允许一个 runtime worker。开发时若要使用 Compose 的数据库，应先停止 Compose 中的 `agent-runtime`；独立验证可使用另建数据库和存储目录，避免与运行中的任务竞争。
