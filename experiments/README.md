# 本机 Agent 恢复与任务评测

这些脚本是独立实验设施，不修改生产图、模型适配或 Java 执行语义。正式执行顺序由后端实验总控安排：缓存 → 秒杀 → 本目录恢复 → 本目录评测。结果目录默认不进入 Git；不要提交含登录令牌的 bootstrap 文件。

## 依赖与隔离协议

复用 Runtime Python 环境，并安装 `python -m pip install -r experiments/requirements.txt`。必须有独立 PostgreSQL/pgvector、Java、MySQL、Redis 和 RocketMQ。Java 的 Runtime URL 必须指向本 CLI 管理的子进程；不能与已有 runtime worker 共享数据库。Java 需要已存在的课程/校区，以及登录用户和工作空间。

所有凭据通过环境变量传递，不放 CLI 参数，也不写结果 manifest：

| 变量 | 用途 |
|---|---|
| `AGENT_EXPERIMENT_ISOLATED=yes` | 确认已选择实验服务 |
| `AGENT_EXPERIMENT_DATABASE_URL` | 独立 PostgreSQL SQLAlchemy URL |
| `AGENT_EXPERIMENT_JAVA_URL` | Java 基地址，无 `/api/v1` 后缀 |
| `AGENT_INTERNAL_TOKEN` | 与实验 Java 一致的内部凭据 |
| `AGENT_EXPERIMENT_AUTH_TOKEN` | Java 登录返回的 Authorization 原值 |
| `AGENT_EXPERIMENT_ACTOR_ID` / `AGENT_EXPERIMENT_WORKSPACE_ID` | 该登录身份及其有权限的空间 |
| `AGENT_EXPERIMENT_RUNTIME_PORT` | 默认 18100；Docker 总控使用 28400 |
| `AGENT_EXPERIMENT_BIND_HOST` | 默认 127.0.0.1；隔离容器间访问设为 0.0.0.0 |
| `AGENT_EXPERIMENT_STORAGE_ROOT` | 独立持久目录，重启之间保留 |
| `AGENT_EXPERIMENT_MYSQL_HOST/PORT/USER/PASSWORD/DATABASE` | 只读业务断言；正式执行应配合 `--require-db-oracle` |
| `DASHSCOPE_API_KEY`、`AI_CHAT_MODEL`、`AI_EMBEDDING_MODEL` | 仅真实模型评测使用，沿用生产配置 |
| `AGENT_EXPERIMENT_REPOSITORY_COMMIT` | 容器没有 git 时，由宿主传入 Agent 仓库提交 SHA |

CLI 只 kill 自己创建且持有句柄的 Runtime 子进程，端口存在监听时拒绝接管。父 CLI 内部的 loopback 代理转发真实 Java 工具请求。故障代理在真实 Java 响应成功后扣住响应、通知控制器 kill 子进程，不伪造业务成功。任务从 Java 公网 API 接收，经真实 outbox/MQ/HTTP 投递给 Runtime；重复命令实验直接重放 HTTP 边界，不声称主动制造 broker 重投。

## 恢复

先执行每类一轮工程预检，再运行正式 20 轮，两个输出目录分开：

```sh
python -m experiments.recovery --rounds 1 --require-db-oracle --output experiments/results/recovery-preflight
python -m experiments.recovery --rounds 20 --require-db-oracle --output experiments/results/recovery-main
```

四个故障：`persisted_queue_kill`、`approval_wait_restart`、`committed_response_lost`、`duplicate_commands`。可用 `--faults` 选择其中几类。均使用 fixture 模型，不测试模型质量。普通课程预约不占试听库存。

`cases.jsonl` 保存每次故障、真实工具回执、PG任务/消息/终态事件数、MySQL预约/申请/订单连接查询结果及逐项断言。报告恢复率分母包含失败；P95仅针对成功样本并显式命名。恢复时间从重启后 `/health` 可用到预期终态，进程启动时间单独统计；重复命令类别测批准到完成。MySQL oracle 未启用时重复/未批准副作用统计为未知，不能写零。

## 50 条完整 Agent 任务评测

`agent_cases.jsonl` 是初始合成基准，包含开发/保留标签、预期工具、禁止工具、预期状态、审批决定、业务副作用及少量 RAG 事实标签。它不是公开 benchmark，也不是生产流量。先冻结保留集，再用开发集调整策略；不要把查看过的失败样本继续当盲测样本。三份合成政策语料在 `corpus/`，没有真实业务承诺。

```sh
# 数据校验不连接服务、不调用模型
python -m experiments.evaluate --validate-only --output experiments/results/validate
# 现有 fixture 能覆盖的确定性工程轨迹；其他样本不得视为自动通过
python -m experiments.evaluate --fixture-smoke --limit 3 --require-db-oracle --output experiments/results/eval-fixture
# 显式允许真实模型，并指定样本数；默认上限 5，可以明确调整
python -m experiments.evaluate --provider bailian --allow-real-model --limit 3 --max-real-cases 3 --require-db-oracle --output experiments/results/eval-real-small
# 按 case ID 选择具有代表性的少量真实样本
python -m experiments.evaluate --provider bailian --allow-real-model --case-ids course_query-01 missing_information-01 reservation_approve-01 --limit 3 --require-db-oracle --output experiments/results/eval-real-selected
```

每例创建独立空间；试听样本自动准备独立活动，避免每人每活动只能参与一次导致样本污染。`trial_future` 按用户结果评测：允许 Agent 直接解释未开场并 `SUCCEEDED`，也允许准备草稿后提交被 Java 拒绝而 `FAILED`；两者都必须说明未开放，并且零业务请求/订单。缺信息样本允许 `WAITING_INPUT` 或完成一次明确询问的 `SUCCEEDED`，不强制某个工具调用顺序。等待补充信息的样本验证后由测试器取消，防止污染后续调度。

实际跑的是完整 `reason → tools → approval/input → …` 图。自动评分核对图状态、工具、数据库终态和显式事实，称 `ruleTaskPassRate`；自由文本的事实准确性和引用语义仍需人工检查，不能称完整语义正确率。fixture 的质量分始终为 null。真实模型费用目前仅统计已有 chat usage；文档/查询 embedding 未被生产Runtime完整归集，完整费用保持 null。

真实批次默认 `--max-estimated-chat-cost-cny 5`，每次开始前按 `AGENT_COST_LIMIT_CNY`（默认 0.10）预留一例预算；已知估算不足或出现未知 chat usage 时不再开始新例，未执行ID单列。该门限是应用层估算，单请求可跨越每run阈值，不能当厂商硬计费上限；embedding费用独立记录。

人工复核最低字段是 `caseId/prompt/expectedFacts/expectedOracle`、`finalRun.answer/error/citations`、`events` 的工具参数和真实工具结果、`approval/approvalFinal`、`businessOracle`，以及 usage/费用。每例JSONL已保存这些字段。评审记录另存 `caseId + factualAccuracy + respectsIntent + citationSupport + businessReceiptMatches + reviewerNotes`，把人工判断与规则业务通过率分列。

评分器 `agent-contract-v2-business-fields` 另从冻结提示中提取显式姓名和唯一11位手机号，核对草稿及实际预约记录，并检查落库课程、校区、姓名、联系方式与已批准参数一致。业务oracle保存 `course/student_name/contact_info/school`、审批 `args/result`，结果包含 `expectedBusinessFields/businessFieldChecks` 供复核。真实模型规则主分纳入这些检查；fixture 固定的测试姓名/号码只作差异诊断，不计语义分，结构烟测可以通过。未明确偏好某课程或校区时不凭空规定选择。

## 设施自检

```sh
python -m unittest experiments.test_harness -v
```

这些自检只用合成数据和本机临时 HTTP 服务，验证“没有真实订单不得判成功”、禁止写操作、失败分母、付费调用 gate 和“真实上游已提交但响应未送达”故障协议。它们不是正式 PostgreSQL/Java 故障实验结果。

## 冻结向量的 RAG 离线对照

`rag_corpus.jsonl` 包含 25 个合成课程、100 份短政策文档，每文档一个固定 chunk。`rag_questions.jsonl` 有 50 题，20 开发、30 保留；开发与保留使用不同课程。标签逐题列出全部相关 documentId/chunkId；多证据问题必须同时找回两份政策才计全部证据覆盖。该集是可控的初始正确性样本，后续还需实际课程文档、同名干扰、无答案问题及人工复核。

```sh
python -m experiments.rag validate
# fixture 仅检查管线，哈希向量成绩不能用于语义检索简历指标
python -m experiments.rag prepare --provider fixture --cache experiments/results/rag-fixture-cache.json
# 真实向量：100 个文档文本 + 50 个独立问题请求；显式计费许可
python -m experiments.rag prepare --provider bailian --allow-real-model --max-texts 150 --cache experiments/results/rag-real-cache.json
# 以下对照完全不调用模型；需要独立 PostgreSQL 环境变量
python -m experiments.rag compare --cache experiments/results/rag-real-cache.json --engine postgres --repeats 5 --output experiments/results/rag-postgres
# 可选纯内存排序微基准，不能称数据库/API 检索延迟
python -m experiments.rag compare --cache experiments/results/rag-real-cache.json --engine memory --output experiments/results/rag-memory
```

prepare 每批/每题保存 `.partial.json`，故障后用 `--resume-prepare` 续跑，已完成缓存不可覆盖。向量缓存固定模型、维度、语料与问题 SHA256、usage 和独立 query embedding 耗时，compare 校验一致性。

对照为 dense-only、中文词项重叠-only、RRF(k=60)。PostgreSQL 模式在 `experiment_rag_<缓存哈希>` schema 使用 pgvector 精确排序，词法部分沿用当前全量候选传回 Python 的实现；不修改生产知识表、不引入 ANN 或 BGE。每题各变体重复测量并打乱变体顺序。报告开发集和保留集各自的 macro Recall@5、MRR@5、全部证据覆盖率、retrieval P95、query embedding P95；检索耗时不包含 embedding、生成、HTTP和鉴权，100-chunk 结果不能代表大语料容量。
