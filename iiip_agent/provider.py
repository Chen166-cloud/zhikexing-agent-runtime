"""百炼兼容接口：有界重试、流式工具调用和实际 usage 统计。"""

import asyncio
import hashlib
import json
import math
import random
import re
from collections.abc import Awaitable, Callable

import httpx
import jieba

from .config import Settings


class ModelError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def lexical_terms(content: str) -> list[str]:
    return [x.lower() for x in jieba.lcut(content) if re.search(r"[\w\u4e00-\u9fff]", x)]


def price_usage(usage: dict, model: str) -> dict:
    """价格快照：百炼北京，2026-09-18；未知用量保持未知。"""
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    result = {
        "model": model,
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "totalTokens": usage.get("total_tokens"),
        "estimatedCostCny": None,
        "priceVersion": "bailian-beijing-2026-09-18",
    }
    if input_tokens is not None and output_tokens is not None and model == "qwen3.7-flash":
        rate = (
            (0.2, 0.8)
            if input_tokens <= 32000
            else (0.6, 2.4) if input_tokens <= 256000 else (1.2, 4.8)
        )
        result["estimatedCostCny"] = round(
            (input_tokens * rate[0] + output_tokens * rate[1]) / 1_000_000, 8
        )
        result["totalTokens"] = input_tokens + output_tokens
    return result


class ModelProvider:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(90, connect=10),
            headers={"Authorization": f"Bearer {settings.api_key}"},
        )

    async def close(self):
        await self.client.aclose()

    async def chat(
        self, messages: list[dict], tools: list[dict], emit: Callable[[str], Awaitable[None]]
    ):
        if self.settings.provider == "fixture":
            return await self.fixture_chat(messages, tools, emit)
        payload = {
            "model": self.settings.chat_model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": 2048,
            "enable_thinking": False,
        }
        if tools:
            payload["tools"] = tools
        for attempt in range(3):
            content, calls, usage = "", {}, {}
            received = False
            try:
                async with self.client.stream(
                    "POST", f"{self.settings.api_base}/chat/completions", json=payload
                ) as response:
                    if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                        delay = response.headers.get("Retry-After", "")
                        await response.aread()
                        await asyncio.sleep(
                            min(float(delay), 15)
                            if delay.isdigit()
                            else 2**attempt + random.random()
                        )
                        continue
                    if response.is_error:
                        raise ModelError(
                            f"MODEL_HTTP_{response.status_code}",
                            f"百炼接口返回 HTTP {response.status_code}，请检查模型权限、参数或额度",
                        )
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        item = json.loads(data)
                        if item.get("usage"):
                            usage = item["usage"]
                        for choice in item.get("choices", []):
                            delta = choice.get("delta", {})
                            if delta.get("content"):
                                received = True
                                content += delta["content"]
                                await emit(delta["content"])
                            for fragment in delta.get("tool_calls", []):
                                received = True
                                call = calls.setdefault(
                                    fragment["index"],
                                    {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    },
                                )
                                if fragment.get("id"):
                                    call["id"] = fragment["id"]
                                function = fragment.get("function", {})
                                call["function"]["name"] += function.get("name", "")
                                call["function"]["arguments"] += function.get("arguments", "")
                message = {"role": "assistant", "content": content}
                if calls:
                    message["tool_calls"] = list(calls.values())
                return message, price_usage(usage, self.settings.chat_model)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if received or attempt == 2:
                    raise ModelError(
                        "MODEL_CONNECTION_FAILED", "模型连接中断，请重试当前任务"
                    ) from exc
                await asyncio.sleep(2**attempt + random.random())
        raise ModelError("MODEL_RETRY_EXHAUSTED", "模型暂不可用，已停止重试")

    async def embed(self, texts: list[str]) -> tuple[list[list[float]], int | None]:
        if self.settings.provider == "fixture":
            vectors = []
            for content in texts:
                vector = [0.0] * self.settings.dimensions
                for term in lexical_terms(content):
                    index = int(hashlib.sha256(term.encode()).hexdigest()[:8], 16) % len(vector)
                    vector[index] += 1
                norm = math.sqrt(sum(x * x for x in vector)) or 1
                vectors.append([x / norm for x in vector])
            return vectors, 0
        payload = {
            "model": self.settings.embedding_model,
            "input": texts,
            "dimensions": self.settings.dimensions,
            "encoding_format": "float",
        }
        for attempt in range(3):
            try:
                response = await self.client.post(
                    f"{self.settings.api_base}/embeddings", json=payload
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                    await asyncio.sleep(2**attempt + random.random())
                    continue
                if response.is_error:
                    raise ModelError(
                        f"EMBEDDING_HTTP_{response.status_code}",
                        f"向量化接口返回 HTTP {response.status_code}",
                    )
                data = response.json()
                vectors = [
                    x["embedding"] for x in sorted(data["data"], key=lambda item: item["index"])
                ]
                if len(vectors) != len(texts) or any(
                    len(x) != self.settings.dimensions for x in vectors
                ):
                    raise ModelError(
                        "EMBEDDING_DIMENSION_MISMATCH", "向量数量或维度与索引配置不一致"
                    )
                return vectors, data.get("usage", {}).get("total_tokens")
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt == 2:
                    raise ModelError("EMBEDDING_CONNECTION_FAILED", "向量化连接失败") from exc
                await asyncio.sleep(2**attempt)
        raise ModelError("EMBEDDING_RETRY_EXHAUSTED", "向量化重试次数已用尽")

    async def fixture_chat(self, messages, tools, emit):
        """显式测试模式才启用；业务工具仍经过真实 Java 接口。"""
        input_text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        tool_messages = [m for m in messages if m["role"] == "tool"]
        names = [m.get("name") for m in tool_messages]
        call = None
        if tools and any(word in input_text for word in ("试听", "秒杀")):
            action_match = re.search(
                r"(?:actionId|动作编号)[：:=\s]+([a-zA-Z0-9_-]{1,64})", input_text
            )
            if action_match and "query_trial_claim" not in names:
                call = ("query_trial_claim", {"actionId": action_match[1]})
            elif "query_trial_campaigns" not in names:
                call = ("query_trial_campaigns", {})
            elif "draft_trial_claim" not in names and any(
                word in input_text for word in ("抢", "领取", "申请", "报名", "参加", "预约")
            ):
                campaigns = json.loads(
                    next(
                        m["content"]
                        for m in tool_messages
                        if m.get("name") == "query_trial_campaigns"
                    )
                )
                if isinstance(campaigns, list) and campaigns:
                    campaign_id = campaigns[0].get("id", campaigns[0].get("campaignId"))
                    if campaign_id is not None:
                        call = ("draft_trial_claim", {"campaignId": str(campaign_id)})
        elif tools and "预约" in input_text and "业务执行结果" not in input_text:
            if "search_courses" not in names:
                call = ("search_courses", {})
            elif "list_campuses" not in names:
                call = ("list_campuses", {})
            elif "draft_reservation" not in names:
                courses = json.loads(
                    next(m["content"] for m in tool_messages if m.get("name") == "search_courses")
                )
                campuses = json.loads(
                    next(m["content"] for m in tool_messages if m.get("name") == "list_campuses")
                )
                if courses and campuses:
                    call = (
                        "draft_reservation",
                        {
                            "courseId": courses[0]["id"],
                            "schoolId": campuses[0]["id"],
                            "studentName": "测试学员",
                            "contactInfo": "13800000000",
                            "remark": "fixture 隔离测试",
                        },
                    )
        elif tools and "search_knowledge" not in names:
            call = ("search_knowledge", {"query": input_text})
        usage = {
            "model": "fixture",
            "inputTokens": 0,
            "outputTokens": 0,
            "totalTokens": 0,
            "estimatedCostCny": 0,
            "provider": "fixture",
        }
        if call:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"fixture_{len(names)}",
                        "type": "function",
                        "function": {
                            "name": call[0],
                            "arguments": json.dumps(call[1], ensure_ascii=False),
                        },
                    }
                ],
            }, usage
        evidence = next(
            (
                json.loads(m["content"])
                for m in reversed(tool_messages)
                if m.get("name") == "search_knowledge"
            ),
            [],
        )
        answer = "【fixture 测试响应】" + (
            evidence[0]["content"] + f" [E1]"
            if evidence
            else "已完成本次模拟回答。" + input_text[:120]
        )
        for offset in range(0, len(answer), 20):
            await emit(answer[offset : offset + 20])
        return {"role": "assistant", "content": answer}, usage
