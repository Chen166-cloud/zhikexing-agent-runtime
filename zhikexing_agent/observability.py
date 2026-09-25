"""模型请求的精简遥测：保留关联编号和用量，不采集提示词、回答或异常正文。"""

import base64
import json
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer


def configure_telemetry(service_name: str) -> tuple[Tracer, Callable[[], None]]:
    """返回专用 tracer 和关闭函数；未配置导出目标时不创建后台线程或网络请求。"""
    collector = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "").strip()
    if not collector:
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip().rstrip("/")
        if endpoint:
            collector = endpoint if endpoint.endswith("/v1/traces") else endpoint + "/v1/traces"

    langfuse_url = os.getenv("LANGFUSE_BASE_URL", "").strip().rstrip("/")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    if (public_key or secret_key) and not (langfuse_url and public_key and secret_key):
        raise ValueError("Langfuse 遥测需要同时配置 BASE_URL、PUBLIC_KEY 和 SECRET_KEY")
    langfuse_enabled = bool(langfuse_url and public_key and secret_key)
    if not collector and not langfuse_enabled:
        return trace.NoOpTracerProvider().get_tracer(service_name), lambda: None

    # 使用实例自己的 provider，不覆盖其它库的全局 OpenTelemetry 配置。
    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name}), shutdown_on_exit=False
    )
    if collector:
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=collector, timeout=5))
        )
    if langfuse_enabled:
        credentials = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        exporter = OTLPSpanExporter(
            endpoint=langfuse_url + "/api/public/otel/v1/traces",
            headers={"Authorization": "Basic " + credentials, "x-langfuse-ingestion-version": "4"},
            timeout=5,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider.get_tracer(service_name), provider.shutdown


@contextmanager
def generation_span(
    tracer: Tracer,
    *,
    run_id: str,
    workspace_id: str,
    provider: str,
    model: str,
    conversation_id: str | None = None,
) -> Iterator[Span]:
    """包围一次真实模型调用；同样适用于 async 函数中的 with 块。"""
    attributes = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": provider,
        "gen_ai.request.model": model,
        "langfuse.observation.type": "generation",
        "langfuse.observation.model.name": model,
        "langfuse.trace.name": "agent.run",
        "langfuse.trace.metadata.run_id": run_id,
        "langfuse.trace.metadata.workspace_id": workspace_id,
        "zhikexing.run.id": run_id,
        "zhikexing.workspace.id": workspace_id,
    }
    if conversation_id:
        attributes["gen_ai.conversation.id"] = conversation_id
        attributes["langfuse.session.id"] = conversation_id
    with tracer.start_as_current_span(
        "chat " + model,
        kind=SpanKind.CLIENT,
        attributes=attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except BaseException as exc:
            # 异常消息可能带用户文本或 HTTP 内容，只记录异常类别。
            span.set_attribute("error.type", type(exc).__name__)
            span.set_attribute("langfuse.observation.level", "ERROR")
            span.set_status(Status(StatusCode.ERROR))
            raise


def record_generation_usage(span: Span, usage: Mapping[str, object]) -> None:
    """接收 ModelProvider 的用量对象；未知值保持缺失，不转成零。"""
    details = {}
    for source, target in (
        ("inputTokens", "input"),
        ("outputTokens", "output"),
        ("totalTokens", "total"),
    ):
        value = usage.get(source)
        if value is not None:
            details[target] = int(value)
    if details:
        span.set_attribute("langfuse.observation.usage_details", json.dumps(details))
    if "input" in details:
        span.set_attribute("gen_ai.usage.input_tokens", details["input"])
    if "output" in details:
        span.set_attribute("gen_ai.usage.output_tokens", details["output"])
    if usage.get("model"):
        span.set_attribute("gen_ai.response.model", str(usage["model"]))
    if usage.get("estimatedCostCny") is not None:
        # Langfuse 原生成本字段采用美元，人民币估算只能写带币种的自定义字段。
        span.set_attribute(
            "langfuse.observation.metadata.estimated_cost_cny", float(usage["estimatedCostCny"])
        )
        span.set_attribute("langfuse.observation.metadata.cost_currency", "CNY")
    if usage.get("priceVersion"):
        span.set_attribute(
            "langfuse.observation.metadata.price_version", str(usage["priceVersion"])
        )
