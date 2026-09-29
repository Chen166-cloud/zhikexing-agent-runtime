"""模型只能选择这里公开的业务工具，写操作由 Java 审批和提交。"""

import httpx

from .config import Settings
from .provider import ModelError


def function(name, description, properties, required=()):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


TOOLS = [
    function(
        "search_courses",
        "查询真实课程；price为人民币元（不除以100），duration为天，edu为最低学历要求；以工具结果为准",
        {
            "keyword": {"type": "string"},
            "type": {"type": "string"},
            "edu": {"type": "integer"},
            "sortBy": {"type": "string", "enum": ["price", "duration", "id"]},
            "ascending": {"type": "boolean"},
        },
    ),
    function("list_campuses", "查询实际校区及城市", {}),
    function(
        "search_knowledge",
        "检索当前工作空间已授权知识库，返回可引用证据",
        {"query": {"type": "string"}},
        ["query"],
    ),
    function(
        "draft_reservation",
        "保存预约草稿并等待用户点击批准；此工具不会提交预约。必须先查询课程和校区并取得用户姓名、联系方式",
        {
            "courseId": {"type": "string"},
            "schoolId": {"type": "string"},
            "studentName": {"type": "string"},
            "contactInfo": {"type": "string"},
            "remark": {"type": "string"},
        },
        ["courseId", "schoolId", "studentName", "contactInfo"],
    ),
    function(
        "query_trial_campaigns",
        "查询免费试听活动、开抢窗口和状态；查询与准备草稿都不锁定名额，提交时由服务端校验",
        {},
    ),
    function(
        "draft_trial_claim",
        "准备免费试听抢名额草稿并等待用户批准；仅在用户明确要申请时调用，不提交、不预占名额，费用为0",
        {"campaignId": {"type": "string", "minLength": 1, "maxLength": 64}},
        ["campaignId"],
    ),
    function(
        "query_trial_claim",
        "用之前返回的actionId查询本人免费试听申请；只有SUCCEEDED及真实orderId代表成功，PENDING/RESERVED仍在处理中",
        {"actionId": {"type": "string", "minLength": 1, "maxLength": 64}},
        ["actionId"],
    ),
    function(
        "ask_user",
        "缺少必需信息时暂停任务并请求用户补充",
        {"question": {"type": "string"}},
        ["question"],
    ),
]


def trial_result(value: dict, action_id: str) -> dict:
    """只将 Java 查询到的业务状态当成事实，排队或预留不等于订单成功。"""
    summaries = {
        "PENDING": "申请已受理，仍在处理中，尚未确认获得名额。可用 actionId 继续查询。",
        "RESERVED": "名额已预留，订单仍在处理中，尚未确认成功。可用 actionId 继续查询。",
        "SUCCEEDED": "免费试听订单已成功创建，请以真实 orderId 为凭据。",
        "REJECTED": "申请未成功，未获得免费试听名额，请查看业务原因。",
    }
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("status"), str)
        or value["status"] not in summaries
    ):
        raise ModelError("INVALID_TRIAL_RESULT", "试听申请接口返回了未知状态，请稍后重新查询")
    if value.get("actionId") not in (None, action_id):
        raise ModelError("INVALID_TRIAL_RESULT", "试听申请结果与动作编号不一致")
    if value["status"] == "SUCCEEDED" and (
        not isinstance(value.get("orderId"), str) or not value["orderId"].strip()
    ):
        raise ModelError("INVALID_TRIAL_RESULT", "试听申请成功状态缺少订单凭据，请稍后重新查询")
    return {
        **value,
        "actionId": action_id,
        "confirmed": value["status"] == "SUCCEEDED",
        "summary": summaries[value["status"]],
    }


class ToolClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = httpx.AsyncClient(base_url=settings.java_url, timeout=20)

    async def call(self, run, method: str, path: str, payload=None):
        headers = {
            "X-Internal-Token": self.settings.internal_token,
            "X-Actor-Id": run.actor_id,
            "X-Workspace-Id": run.workspace_id,
            "X-Run-Id": run.id,
        }
        response = await self.client.request(
            method, "/internal/v1/tools" + path, headers=headers, json=payload
        )
        if response.is_error:
            # 只返回约定的业务错误，不把响应头、凭据或整段异常响应放进模型。
            try:
                error = response.json()
            except ValueError:
                error = {}
            raise ModelError(
                error.get("code", f"TOOL_HTTP_{response.status_code}"),
                error.get("message", "业务工具调用失败"),
            )
        return response.json()

    async def close(self):
        await self.client.aclose()
