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
        "查询真实课程；金额、学历要求以工具结果为准",
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
        "ask_user",
        "缺少必需信息时暂停任务并请求用户补充",
        {"question": {"type": "string"}},
        ["question"],
    ),
]


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
