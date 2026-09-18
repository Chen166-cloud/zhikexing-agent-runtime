"""内部 HTTP 边界；Java 注入可信身份，所有资源查询再次限定空间和归属。"""

import asyncio
import hashlib
import hmac
import json
import mimetypes
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from prometheus_client import CollectorRegistry, Gauge, CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import delete, func, select, text

from .config import Settings
from .db import (
    Chunk,
    Conversation,
    Database,
    Document,
    DocumentVersion,
    Evaluation,
    KnowledgeBase,
    Message,
    Run,
    RunEvent,
    new_id,
    now,
)
from .runtime import AgentRuntime, TERMINAL


class ConversationCreate(BaseModel):
    title: str = Field(default="新对话", min_length=1, max_length=160)
    workspaceId: str | None = None


class RunCreate(BaseModel):
    runId: str = Field(min_length=1, max_length=64)
    conversationId: str = Field(min_length=1, max_length=64)
    input: str = Field(min_length=1, max_length=12000)
    knowledgeBaseIds: list[str] = Field(default_factory=list, max_length=20)
    mode: str = "agent"
    clientRequestId: str | None = None
    workspaceId: str | None = None
    actorId: str | None = None


class NameCreate(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    workspaceId: str | None = None


class InputCreate(BaseModel):
    input: str = Field(min_length=1, max_length=12000)
    clientRequestId: str = Field(min_length=1, max_length=100)
    workspaceId: str | None = None


class EvaluationCase(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    expectedText: str = Field(default="", max_length=4000)
    expectedDocumentId: str | None = None


class EvaluationCreate(BaseModel):
    knowledgeBaseId: str
    cases: list[EvaluationCase] = Field(min_length=1, max_length=100)
    workspaceId: str | None = None


def run_view(run: Run):
    return {
        "runId": run.id,
        "conversationId": run.conversation_id,
        "workspaceId": run.workspace_id,
        "status": run.status,
        "input": run.input,
        "answer": run.answer,
        "error": run.error,
        "usage": run.usage,
        "steps": run.steps,
        "citations": run.citations,
        "createdAt": run.created_at,
        "updatedAt": run.updated_at,
    }


def document_view(doc: Document):
    return {
        "id": doc.id,
        "title": doc.title,
        "workspaceId": doc.workspace_id,
        "knowledgeBaseId": doc.knowledge_base_id,
        "status": doc.status,
        "version": doc.active_version,
        "pendingVersion": doc.pending_version,
        "chunkCount": doc.chunk_count,
        "error": doc.error,
        "createdAt": doc.created_at,
    }


def evaluation_view(evaluation: Evaluation):
    return {
        "id": evaluation.id,
        "workspaceId": evaluation.workspace_id,
        "knowledgeBaseId": evaluation.knowledge_base_id,
        "status": evaluation.status,
        "summary": evaluation.summary,
        "results": evaluation.results,
        "createdAt": evaluation.created_at,
    }


def create_app(settings: Settings | None = None):
    settings = settings or Settings()
    settings.validate()
    db = Database(settings.database_url)
    runtime = AgentRuntime(settings, db)
    conversation_locks: dict[str, asyncio.Lock] = {}

    @asynccontextmanager
    async def lifespan(app):
        await db.initialize()
        await runtime.start()
        app.state.db, app.state.runtime = db, runtime
        try:
            yield
        finally:
            await runtime.close()
            await db.close()

    app = FastAPI(title="IIIP Agent Runtime", version="1.0.0", lifespan=lifespan)

    async def scope(
        x_internal_token: str = Header(default=""),
        x_actor_id: str = Header(default=""),
        x_workspace_id: str = Header(default=""),
    ):
        if not hmac.compare_digest(x_internal_token, settings.internal_token):
            raise HTTPException(401, "内部服务身份无效")
        if not x_actor_id or not x_workspace_id or len(x_actor_id) > 64 or len(x_workspace_id) > 64:
            raise HTTPException(400, "缺少有效的调用身份与工作空间")
        return x_actor_id, x_workspace_id

    Scope = Annotated[tuple[str, str], Depends(scope)]

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": f"HTTP_{exc.status_code}", "message": str(exc.detail)},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        fields = [".".join(str(part) for part in error["loc"]) for error in exc.errors()]
        return JSONResponse(
            status_code=422,
            content={"code": "INVALID_ARGUMENTS", "message": "参数格式有误：" + "、".join(fields)},
        )

    async def owned(session, entity, identifier, identity, private=False, lock=False):
        obj = await session.get(entity, identifier, with_for_update=lock)
        if (
            not obj
            or obj.workspace_id != identity[1]
            or (private and obj.actor_id != identity[0])
            or getattr(obj, "deleted", False)
        ):
            raise HTTPException(404, "资源不存在或无权访问")
        return obj

    @app.get("/health")
    async def health():
        async with db.session() as session:
            await session.execute(text("SELECT 1"))
        return {
            "status": "UP",
            "provider": settings.provider,
            "chatModel": settings.chat_model,
            "embeddingModel": settings.embedding_model,
            "dimensions": settings.dimensions,
            "storage": "postgresql" if db.postgres else "sqlite-local",
            "worker": "single-worker",
        }

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics():
        registry = CollectorRegistry()
        run_count = Gauge(
            "iiip_agent_runs", "运行状态数量", ["status", "provider"], registry=registry
        )
        document_count = Gauge(
            "iiip_agent_documents", "文档入库状态数量", ["status"], registry=registry
        )
        active_jobs = Gauge("iiip_agent_active_jobs", "当前后台作业数量", registry=registry)
        async with db.session() as session:
            counts = (
                await session.execute(select(Run.status, func.count()).group_by(Run.status))
            ).all()
            documents = (
                await session.execute(
                    select(Document.status, func.count())
                    .where(Document.deleted.is_(False))
                    .group_by(Document.status)
                )
            ).all()
        for status, count in counts:
            run_count.labels(status=status, provider=settings.provider).set(count)
        for status, count in documents:
            document_count.labels(status=status).set(count)
        active_jobs.set(sum(not task.done() for task in runtime.tasks.values()))
        return Response(generate_latest(registry), headers={"Content-Type": CONTENT_TYPE_LATEST})

    @app.get("/internal/v1/conversations")
    async def conversations(identity: Scope):
        async with db.session() as session:
            rows = (
                await session.scalars(
                    select(Conversation)
                    .where(
                        Conversation.workspace_id == identity[1],
                        Conversation.actor_id == identity[0],
                    )
                    .order_by(Conversation.created_at.desc())
                )
            ).all()
        return [
            {
                "id": row.id,
                "workspaceId": row.workspace_id,
                "title": row.title,
                "createdAt": row.created_at,
            }
            for row in rows
        ]

    @app.post("/internal/v1/conversations", status_code=201)
    async def create_conversation(body: ConversationCreate, identity: Scope):
        async with db.session() as session:
            conversation = Conversation(
                workspace_id=identity[1], actor_id=identity[0], title=body.title
            )
            session.add(conversation)
            await session.commit()
        return {
            "id": conversation.id,
            "workspaceId": identity[1],
            "title": conversation.title,
            "createdAt": conversation.created_at,
        }

    @app.get("/internal/v1/conversations/{identifier}")
    async def conversation(identifier: str, identity: Scope):
        async with db.session() as session:
            item = await owned(session, Conversation, identifier, identity, private=True)
            rows = (
                await session.scalars(
                    select(Message)
                    .where(Message.conversation_id == identifier)
                    .order_by(Message.created_at)
                )
            ).all()
        return {
            "id": item.id,
            "workspaceId": item.workspace_id,
            "title": item.title,
            "createdAt": item.created_at,
            "messages": [
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "runId": row.run_id,
                    "createdAt": row.created_at,
                    "citations": row.citations,
                }
                for row in rows
            ],
        }

    @app.post("/internal/v1/runs", status_code=202)
    async def create_run(
        body: RunCreate, identity: Scope, x_command_delivery: bool = Header(default=False)
    ):
        if body.mode not in {"agent", "chat"}:
            raise HTTPException(400, "不支持的任务模式")
        normalized = {
            "conversationId": body.conversationId,
            "input": body.input,
            "knowledgeBaseIds": sorted(body.knowledgeBaseIds),
            "mode": body.mode,
        }
        digest = hashlib.sha256(
            json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        lock = conversation_locks.setdefault(body.conversationId, asyncio.Lock())
        async with lock, db.session() as session:
            await owned(
                session, Conversation, body.conversationId, identity, private=True, lock=True
            )
            existing = await session.get(Run, body.runId)
            if existing:
                await owned(session, Run, body.runId, identity, private=True)
                if existing.request_hash != digest:
                    raise HTTPException(409, "相同请求标识不能提交不同内容")
                return run_view(existing)
            active = (
                await session.scalars(
                    select(Run).where(
                        Run.conversation_id == body.conversationId, Run.status.not_in(TERMINAL)
                    )
                )
            ).first()
            if active and not x_command_delivery:
                raise HTTPException(409, "当前会话仍有活动任务，请先完成或取消")
            rejection = (
                {
                    "code": "CONVERSATION_BUSY",
                    "message": "投递时会话已有活动任务，请等待其结束后重试",
                }
                if active
                else None
            )
            for kb_id in body.knowledgeBaseIds:
                try:
                    await owned(session, KnowledgeBase, kb_id, identity)
                except HTTPException:
                    if not x_command_delivery:
                        raise
                    rejection = {
                        "code": "KNOWLEDGE_UNAVAILABLE",
                        "message": "投递时所选知识库已不可用，请重新选择知识库",
                    }
            run = Run(
                id=body.runId,
                conversation_id=body.conversationId,
                workspace_id=identity[1],
                actor_id=identity[0],
                request_hash=digest,
                input=body.input,
                mode=body.mode,
                knowledge_base_ids=body.knowledgeBaseIds,
                usage={"configuration": runtime.configuration()},
            )
            if rejection:
                # Java 已向用户接受的命令必须可查询；业务拒绝也先落库，再向队列 ACK。
                run.status, run.error, run.event_seq = "FAILED", rejection, 1
            session.add(run)
            await session.flush()
            if rejection:
                session.add(RunEvent(run_id=run.id, seq=1, type="run.failed", data=rejection))
            else:
                session.add(
                    Message(
                        conversation_id=body.conversationId,
                        run_id=body.runId,
                        message_key="input",
                        role="user",
                        content=body.input,
                    )
                )
            await session.commit()
        return run_view(run)

    @app.get("/internal/v1/runs")
    async def runs(identity: Scope):
        async with db.session() as session:
            rows = (
                await session.scalars(
                    select(Run)
                    .where(Run.workspace_id == identity[1], Run.actor_id == identity[0])
                    .order_by(Run.created_at.desc())
                    .limit(100)
                )
            ).all()
        return [run_view(row) for row in rows]

    @app.get("/internal/v1/runs/{identifier}")
    async def run(identifier: str, identity: Scope):
        async with db.session() as session:
            return run_view(await owned(session, Run, identifier, identity, private=True))

    @app.get("/internal/v1/runs/{identifier}/events")
    async def events(
        identifier: str,
        request: Request,
        identity: Scope,
        after: int = 0,
        last_event_id: str | None = Header(default=None),
    ):
        async with db.session() as session:
            await owned(session, Run, identifier, identity, private=True)
        cursor = max(after, int(last_event_id) if last_event_id and last_event_id.isdigit() else 0)

        async def stream():
            nonlocal cursor
            heartbeat = time.monotonic()
            while not await request.is_disconnected():
                async with db.session() as session:
                    current = await session.get(Run, identifier)
                    rows = (
                        await session.scalars(
                            select(RunEvent)
                            .where(RunEvent.run_id == identifier, RunEvent.seq > cursor)
                            .order_by(RunEvent.seq)
                            .limit(100)
                        )
                    ).all()
                for row in rows:
                    item = {
                        "runId": identifier,
                        "seq": row.seq,
                        "type": row.type,
                        "data": row.data,
                        "createdAt": row.created_at,
                    }
                    yield f"id: {row.seq}\nevent: {row.type}\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"
                    cursor = row.seq
                if current.status in TERMINAL and cursor >= current.event_seq:
                    # 终态与终态事件可能差一个事务，短等一次后以数据库快照为准。
                    await asyncio.sleep(0.2)
                    async with db.session() as session:
                        latest = await session.get(Run, identifier)
                    if latest.event_seq <= cursor:
                        return
                if time.monotonic() - heartbeat >= 10:
                    yield ": heartbeat\n\n"
                    heartbeat = time.monotonic()
                await asyncio.sleep(0.15)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/internal/v1/runs/{identifier}/cancel")
    async def cancel(identifier: str, identity: Scope):
        async with db.session() as session:
            item = await owned(session, Run, identifier, identity, private=True, lock=True)
            if item.status in TERMINAL:
                return run_view(item)
            item.status, item.updated_at = "CANCELLED", now()
            item.event_seq += 1
            session.add(
                RunEvent(
                    run_id=identifier,
                    seq=item.event_seq,
                    type="run.cancelled",
                    data={"status": "CANCELLED"},
                )
            )
            await session.commit()
        task = runtime.tasks.get("run:" + identifier)
        if task:
            task.cancel()
        return run_view(item)

    @app.post("/internal/v1/runs/{identifier}/retry")
    async def retry(identifier: str, identity: Scope):
        async with db.session() as session:
            original = await owned(session, Run, identifier, identity, private=True)
        lock = conversation_locks.setdefault(original.conversation_id, asyncio.Lock())
        async with lock:
            return await retry_locked(identifier, identity)

    async def retry_locked(identifier, identity):
        async with db.session() as session:
            item = await owned(session, Run, identifier, identity, private=True, lock=True)
            await owned(
                session, Conversation, item.conversation_id, identity, private=True, lock=True
            )
            if item.status not in {"FAILED", "TIMED_OUT"}:
                raise HTTPException(409, "只有失败或超时任务可以重试")
            if (
                item.steps >= settings.max_steps
                or (item.usage.get("totalTokens") or 0) >= settings.max_tokens
            ):
                raise HTTPException(409, "当前任务预算已用尽，请缩小问题后新建任务")
            if (item.usage.get("estimatedCostCny") or 0) >= settings.max_cost_cny:
                raise HTTPException(409, "当前任务费用预算已用尽，请缩小问题后新建任务")
            active = (
                await session.scalars(
                    select(Run).where(
                        Run.conversation_id == item.conversation_id,
                        Run.id != identifier,
                        Run.status.not_in(TERMINAL),
                    )
                )
            ).first()
            if active:
                raise HTTPException(409, "会话已有其他活动任务")
            for kb_id in item.knowledge_base_ids:
                await owned(session, KnowledgeBase, kb_id, identity)
            initial_message = (
                await session.scalars(
                    select(Message).where(
                        Message.run_id == identifier, Message.message_key == "input"
                    )
                )
            ).first()
            if not initial_message:
                # 投递时被拒绝的输入尚未进入对话，重试获准后才加入，避免影响正在执行的任务。
                session.add(
                    Message(
                        conversation_id=item.conversation_id,
                        run_id=identifier,
                        message_key="input",
                        role="user",
                        content=item.input,
                    )
                )
            item.status, item.error, item.updated_at = "QUEUED", None, now()
            await session.commit()
        return run_view(item)

    @app.post("/internal/v1/runs/{identifier}/inputs")
    async def inputs(identifier: str, body: InputCreate, identity: Scope):
        async with db.session() as session:
            item = await owned(session, Run, identifier, identity, private=True, lock=True)
            previous = (
                await session.scalars(
                    select(Message).where(
                        Message.run_id == identifier,
                        Message.message_key == "input:" + body.clientRequestId,
                    )
                )
            ).first()
            if previous:
                if previous.content != body.input:
                    raise HTTPException(409, "同一补充请求标识对应不同内容")
                return run_view(item)
            if item.status != "WAITING_INPUT":
                raise HTTPException(409, "任务未等待补充信息；修改待审批草稿请先拒绝后重新发起")
            if item.resume_input is not None:
                raise HTTPException(409, "上一条补充信息正在恢复执行，请等待处理完成")
            item.resume_input = {"input": body.input, "clientRequestId": body.clientRequestId}
            session.add(
                Message(
                    conversation_id=item.conversation_id,
                    run_id=identifier,
                    message_key="input:" + body.clientRequestId,
                    role="user",
                    content=body.input,
                )
            )
            await session.commit()
        return run_view(item)

    @app.get("/internal/v1/knowledge-bases")
    async def knowledge_bases(identity: Scope):
        async with db.session() as session:
            rows = (
                await session.scalars(
                    select(KnowledgeBase).where(KnowledgeBase.workspace_id == identity[1])
                )
            ).all()
        return [{"id": row.id, "name": row.name, "workspaceId": row.workspace_id} for row in rows]

    @app.post("/internal/v1/knowledge-bases", status_code=201)
    async def create_knowledge_base(body: NameCreate, identity: Scope):
        async with db.session() as session:
            kb = KnowledgeBase(name=body.name, workspace_id=identity[1])
            session.add(kb)
            await session.commit()
        return {"id": kb.id, "name": kb.name, "workspaceId": kb.workspace_id}

    @app.get("/internal/v1/documents")
    async def documents(identity: Scope, knowledgeBaseId: str | None = None):
        async with db.session() as session:
            query = select(Document).where(
                Document.workspace_id == identity[1], Document.deleted.is_(False)
            )
            if knowledgeBaseId:
                await owned(session, KnowledgeBase, knowledgeBaseId, identity)
                query = query.where(Document.knowledge_base_id == knowledgeBaseId)
            rows = (await session.scalars(query.order_by(Document.created_at.desc()))).all()
        return [document_view(row) for row in rows]

    @app.post("/internal/v1/documents", status_code=202)
    async def upload_document(
        identity: Scope,
        knowledgeBaseId: Annotated[str, Form()],
        file: Annotated[UploadFile, File()],
        documentId: Annotated[str | None, Form()] = None,
    ):
        async with db.session() as session:
            await owned(session, KnowledgeBase, knowledgeBaseId, identity)
        title = Path(file.filename or "document.pdf").name
        if len(title) > 255 or Path(title).suffix.lower() not in {".pdf", ".txt", ".md"}:
            raise HTTPException(400, "请上传 PDF、TXT 或 Markdown 文档")
        content = await file.read(20 * 1024 * 1024 + 1)
        if not content or len(content) > 20 * 1024 * 1024:
            raise HTTPException(413, "文件为空或超过 20 MB")
        if title.lower().endswith(".pdf") and not content.startswith(b"%PDF-"):
            raise HTTPException(400, "文件内容不是有效的 PDF")
        content_hash = hashlib.sha256(content).hexdigest()
        async with db.session() as session:
            if documentId:
                doc = await owned(session, Document, documentId, identity, lock=True)
                if doc.knowledge_base_id != knowledgeBaseId or doc.status not in {
                    "READY",
                    "FAILED",
                }:
                    raise HTTPException(409, "文档不属于该知识库或已有版本正在处理")
                doc.pending_version += 1
                doc.status = "UPLOADED"
            else:
                duplicate = (
                    await session.scalars(
                        select(Document)
                        .join(DocumentVersion, DocumentVersion.document_id == Document.id)
                        .where(
                            Document.workspace_id == identity[1],
                            Document.knowledge_base_id == knowledgeBaseId,
                            Document.deleted.is_(False),
                            DocumentVersion.content_hash == content_hash,
                        )
                    )
                ).first()
                if duplicate:
                    return document_view(duplicate)
                doc = Document(
                    id=new_id(),
                    workspace_id=identity[1],
                    knowledge_base_id=knowledgeBaseId,
                    title=title,
                    pending_version=1,
                )
                session.add(doc)
            key = f"{identity[1]}/{doc.id}/{doc.pending_version}{Path(title).suffix.lower()}"
            await runtime.storage.put(key, content)
            await session.flush()
            session.add(
                DocumentVersion(
                    document_id=doc.id,
                    version=doc.pending_version,
                    object_key=key,
                    content_hash=content_hash,
                    embedding_model=settings.embedding_model,
                )
            )
            await session.commit()
        return document_view(doc)

    @app.post("/internal/v1/documents/{identifier}/retry")
    async def retry_document(identifier: str, identity: Scope):
        async with db.session() as session:
            doc = await owned(session, Document, identifier, identity)
            if doc.status != "FAILED":
                raise HTTPException(409, "只有处理失败的文档可以重试")
            doc.status, doc.error = "UPLOADED", None
            await session.commit()
        return document_view(doc)

    @app.delete("/internal/v1/documents/{identifier}", status_code=202)
    async def delete_document(identifier: str, identity: Scope):
        async with db.session() as session:
            doc = await owned(session, Document, identifier, identity)
            doc.deleted, doc.status = True, "DELETING"
            await session.execute(delete(Chunk).where(Chunk.document_id == identifier))
            await session.commit()
        return {"id": identifier, "status": "DELETING"}

    @app.get("/internal/v1/documents/{identifier}/content")
    async def document_content(identifier: str, identity: Scope):
        async with db.session() as session:
            doc = await owned(session, Document, identifier, identity)
            version = (
                await session.scalars(
                    select(DocumentVersion).where(
                        DocumentVersion.document_id == identifier,
                        DocumentVersion.version == (doc.active_version or doc.pending_version),
                    )
                )
            ).one()
        content = await runtime.storage.get(version.object_key)
        return Response(
            content,
            media_type=mimetypes.guess_type(doc.title)[0] or "application/octet-stream",
            headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
        )

    @app.get("/internal/v1/documents/{identifier}/chunks")
    async def document_chunks(identifier: str, identity: Scope):
        async with db.session() as session:
            doc = await owned(session, Document, identifier, identity)
            rows = (
                await session.scalars(
                    select(Chunk)
                    .where(Chunk.document_id == identifier, Chunk.version == doc.active_version)
                    .order_by(Chunk.ordinal)
                )
            ).all()
        return [{"id": row.id, "page": row.page, "content": row.content} for row in rows]

    @app.post("/internal/v1/evaluations", status_code=202)
    async def create_evaluation(body: EvaluationCreate, identity: Scope):
        async with db.session() as session:
            await owned(session, KnowledgeBase, body.knowledgeBaseId, identity)
            evaluation = Evaluation(
                workspace_id=identity[1],
                actor_id=identity[0],
                knowledge_base_id=body.knowledgeBaseId,
                cases=[item.model_dump() for item in body.cases],
            )
            session.add(evaluation)
            await session.commit()
        return evaluation_view(evaluation)

    @app.get("/internal/v1/evaluations")
    async def evaluations(identity: Scope):
        async with db.session() as session:
            rows = (
                await session.scalars(
                    select(Evaluation)
                    .where(Evaluation.workspace_id == identity[1])
                    .order_by(Evaluation.created_at.desc())
                    .limit(50)
                )
            ).all()
        return [evaluation_view(row) for row in rows]

    @app.get("/internal/v1/evaluations/{identifier}")
    async def evaluation(identifier: str, identity: Scope):
        async with db.session() as session:
            return evaluation_view(await owned(session, Evaluation, identifier, identity))

    @app.get("/internal/v1/metrics")
    async def metrics(identity: Scope):
        async with db.session() as session:
            rows = (
                await session.scalars(
                    select(Run).where(Run.workspace_id == identity[1], Run.actor_id == identity[0])
                )
            ).all()
        statuses = {}
        for row in rows:
            statuses[row.status] = statuses.get(row.status, 0) + 1
        known = [row for row in rows if row.usage.get("estimatedCostCny") is not None]
        return {
            "runCount": len(rows),
            "successRate": statuses.get("SUCCEEDED", 0) / len(rows) if rows else 0,
            "totalTokens": sum(row.usage.get("totalTokens") or 0 for row in rows),
            "estimatedCostCny": (
                sum(row.usage["estimatedCostCny"] for row in known)
                if len(known) == len(rows)
                else None
            ),
            "knownCostCny": sum(row.usage["estimatedCostCny"] for row in known),
            "unknownUsageRuns": len(rows) - len(known),
            "averageLatencyMs": sum(row.latency_ms for row in rows) / len(rows) if rows else 0,
            "statusCounts": statuses,
            "provider": settings.provider,
        }

    return app
