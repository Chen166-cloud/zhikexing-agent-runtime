"""隔离的试听审批/结果回归；不调用计费模型、真实 Java 或共享数据库。"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from zhikexing_agent.config import Settings
from zhikexing_agent.db import Conversation, Database, Message, Run
from zhikexing_agent.provider import ModelError
from zhikexing_agent.runtime import AgentRuntime
from zhikexing_agent.tools import trial_result


class TrialResultTest(unittest.TestCase):
    def test_only_committed_order_is_confirmed(self):
        for status in ("PENDING", "RESERVED", "REJECTED"):
            with self.subTest(status=status):
                result = trial_result({"status": status}, "action-1")
                self.assertFalse(result["confirmed"])
        self.assertTrue(
            trial_result({"status": "SUCCEEDED", "orderId": "order-1"}, "action-1")["confirmed"]
        )

    def test_inconsistent_or_incomplete_result_is_rejected(self):
        for value in (
            {"status": "ACCEPTED"},
            {"status": "SUCCEEDED"},
            {"status": "SUCCEEDED", "orderId": " "},
            {"status": ["SUCCEEDED"]},
            {"status": "PENDING", "actionId": "other-action"},
            None,
        ):
            with self.subTest(value=value), self.assertRaises(ModelError):
                trial_result(value, "action-1")


class TrialGraphTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.directory = tempfile.TemporaryDirectory(prefix="zhikexing-trial-test-")
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.settings = Settings(
            database_url=f"sqlite+aiosqlite:///{root / 'agent.db'}",
            storage_root=root,
            internal_token="isolated-internal-token",
            provider="fixture",
            api_key="",
            max_parallel=0,
        )
        self.db = Database(self.settings.database_url)
        await self.db.initialize()
        self.runtime = AgentRuntime(self.settings, self.db)
        await self.runtime.tools.client.aclose()
        self.runtime.tools.client = httpx.AsyncClient(
            base_url="http://java.test", transport=httpx.MockTransport(self.java_request)
        )
        self.calls = []
        self.statuses = ["SUCCEEDED"]
        self.approvals = {}
        self.requests = {}
        self.draft_calls = []
        self.execute_calls = 0
        self.query_calls = 0
        await self.runtime.start()
        async with self.db.session() as session:
            session.add(
                Conversation(
                    id="conversation-1", workspace_id="space-1", actor_id="user-1", title="试听"
                )
            )
            await session.commit()

    async def asyncTearDown(self):
        await self.runtime.close()
        await self.db.close()

    def result(self, action_id, status):
        return {
            "actionId": action_id,
            "requestId": "request-1",
            "campaignId": "campaign-1",
            "status": status,
            "orderId": "order-1" if status == "SUCCEEDED" else None,
            "reason": "SOLD_OUT" if status == "REJECTED" else None,
        }

    def java_request(self, request):
        self.assertEqual(request.headers["X-Internal-Token"], "isolated-internal-token")
        self.assertEqual(request.headers["X-Actor-Id"], "user-1")
        self.assertEqual(request.headers["X-Workspace-Id"], "space-1")
        self.assertEqual(request.headers["X-Run-Id"], "run-1")
        path = request.url.path.removeprefix("/internal/v1/tools")
        self.calls.append((request.method, path))
        body = json.loads(request.content) if request.content else {}
        if path == "/trial-campaigns":
            return httpx.Response(200, json=[{"id": "campaign-1", "title": "免费试听", "price": 0}])
        if path == "/draft-trial-claim":
            self.draft_calls.append(body)
            approval = {
                "id": "approval-1",
                "actionId": body["actionId"],
                "toolName": "claim_trial",
                "status": "PENDING",
                "args": {"campaignId": body["campaignId"], "title": "免费试听", "price": 0},
            }
            self.approvals.setdefault(body["actionId"], approval)
            return httpx.Response(200, json=self.approvals[body["actionId"]])
        if path == "/execute-trial-claim":
            self.execute_calls += 1
            self.assertEqual(body["approvalId"], "approval-1")
            self.requests.setdefault(body["actionId"], self.result(body["actionId"], "PENDING"))
            return httpx.Response(202, json=self.requests[body["actionId"]])
        if path.startswith("/trial-claims/by-action/"):
            action_id = path.rsplit("/", 1)[-1]
            self.assertIn(action_id, self.requests)
            self.query_calls += 1
            status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            return httpx.Response(200, json=self.result(action_id, status))
        if path == "/courses":
            return httpx.Response(200, json=[{"id": "course-1"}])
        if path == "/campuses":
            return httpx.Response(200, json=[{"id": "school-1"}])
        if path == "/draft-reservation":
            return httpx.Response(
                200, json={"id": "approval-1", "actionId": body["actionId"], "status": "PENDING"}
            )
        if path == "/execute-reservation":
            return httpx.Response(200, json={"reservationId": "unverified-response"})
        if path.startswith("/reservations/by-action/"):
            return httpx.Response(200, json={"reservationId": "verified-reservation"})
        raise AssertionError(f"Unexpected tool request {request.method} {path}")

    async def create_run(self, content="帮我领取免费试听名额"):
        async with self.db.session() as session:
            session.add(
                Run(
                    id="run-1",
                    conversation_id="conversation-1",
                    actor_id="user-1",
                    workspace_id="space-1",
                    request_hash="test-hash",
                    input=content,
                    usage={"configuration": self.runtime.configuration()},
                )
            )
            session.add(
                Message(
                    conversation_id="conversation-1",
                    run_id="run-1",
                    message_key="input",
                    role="user",
                    content=content,
                )
            )
            await session.commit()
        await self.runtime.execute_run("run-1")
        return await self.runtime.get_run("run-1")

    async def state(self):
        return (
            await self.runtime.graph.aget_state({"configurable": {"thread_id": "run-1"}})
        ).values

    async def approve(self, status="APPROVED"):
        await self.runtime.execute_run("run-1", {"status": status, "toolName": "claim_trial"})
        return await self.runtime.get_run("run-1")

    async def test_approval_required_and_success_comes_from_query(self):
        waiting = await self.create_run()
        self.assertEqual(waiting.status, "WAITING_APPROVAL")
        self.assertEqual(self.execute_calls, 0)
        self.assertEqual((await self.state())["pendingApproval"]["toolName"], "claim_trial")
        finished = await self.approve()
        self.assertEqual(finished.status, "SUCCEEDED")
        self.assertIn("order-1", finished.answer)
        self.assertEqual(self.execute_calls, 1)
        self.assertEqual(self.query_calls, 1)
        self.assertEqual(len(self.requests), 1)

    async def test_pending_is_bounded_and_never_claims_success(self):
        self.statuses = ["PENDING", "RESERVED", "RESERVED"]
        await self.create_run()
        finished = await self.approve()
        self.assertIn("仍在处理中", finished.answer)
        self.assertIn("动作编号", finished.answer)
        self.assertNotIn("订单已成功创建", finished.answer)
        self.assertIn("不会撤销", finished.answer)
        self.assertEqual(self.execute_calls, 1)
        self.assertEqual(self.query_calls, 3)
        self.assertFalse((await self.state())["trialOutcome"]["confirmed"])

    async def test_sold_out_reports_rejection_without_resubmission(self):
        self.statuses = ["REJECTED"]
        await self.create_run()
        finished = await self.approve()
        self.assertIn("申请未成功", finished.answer)
        self.assertIn("SOLD_OUT", finished.answer)
        self.assertEqual(self.execute_calls, 1)

    async def test_declined_approval_never_submits(self):
        await self.create_run()
        await self.approve("REJECTED")
        self.assertEqual(self.execute_calls, 0)
        self.assertEqual(self.query_calls, 0)

    async def test_executed_approval_after_restart_only_queries_result(self):
        await self.create_run()
        saved = await self.state()
        action_id = saved["pendingApproval"]["actionId"]
        self.requests[action_id] = self.result(action_id, "PENDING")
        await self.runtime.close()
        previous_client = self.runtime.tools.client
        self.runtime = AgentRuntime(self.settings, self.db)
        await self.runtime.tools.client.aclose()
        self.runtime.tools.client = httpx.AsyncClient(
            base_url="http://java.test", transport=httpx.MockTransport(self.java_request)
        )
        await self.runtime.start()
        finished = await self.approve("EXECUTED")
        self.assertIn("order-1", finished.answer)
        self.assertEqual(self.execute_calls, 0)
        self.assertTrue(previous_client.is_closed)

    async def test_replayed_draft_keeps_same_action_id(self):
        await self.create_run()
        saved = await self.state()
        # 模拟 draft HTTP 成功后、tools checkpoint 写入前崩溃：重放相同模型 tool_call。
        assistant_index = max(i for i, m in enumerate(saved["messages"]) if m.get("tool_calls"))
        replay = {**saved, "messages": saved["messages"][: assistant_index + 1]}
        first = await self.runtime.execute_tools(replay)
        second = await self.runtime.execute_tools(replay)
        self.assertEqual(
            first["pendingApproval"]["actionId"], second["pendingApproval"]["actionId"]
        )
        self.assertEqual(len(self.approvals), 1)
        self.assertEqual(self.execute_calls, 0)

    async def test_followup_query_is_not_cached_or_resubmitted(self):
        self.statuses = ["PENDING", "SUCCEEDED"]
        await self.create_run()
        saved = await self.state()
        action_id = saved["pendingApproval"]["actionId"]
        self.requests[action_id] = self.result(action_id, "PENDING")
        query = {
            "runId": "run-1",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "query-1",
                            "function": {
                                "name": "query_trial_claim",
                                "arguments": json.dumps({"actionId": action_id}),
                            },
                        }
                    ],
                }
            ],
        }
        pending = await self.runtime.execute_tools(query)
        completed = await self.runtime.execute_tools({**query, "readCache": pending["readCache"]})
        self.assertFalse(pending["trialOutcome"]["confirmed"])
        self.assertTrue(completed["trialOutcome"]["confirmed"])
        self.assertEqual(self.query_calls, 2)
        self.assertEqual(self.execute_calls, 0)

    async def test_model_cannot_supply_identity_or_execution_tool(self):
        await self.create_run()
        before = len(self.draft_calls)
        for name, args in (
            ("draft_trial_claim", {"campaignId": "campaign-1", "actorId": "other"}),
            ("query_trial_claim", {"actionId": "../../other"}),
            ("execute_trial_claim", {"actionId": "anything"}),
        ):
            output = await self.runtime.execute_tools(
                {
                    "runId": "run-1",
                    "messages": [
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "invalid-1",
                                    "function": {"name": name, "arguments": json.dumps(args)},
                                }
                            ],
                        }
                    ],
                }
            )
            self.assertIn("error", json.loads(output["messages"][-1]["content"]))
        self.assertEqual(len(self.draft_calls), before)
        self.assertEqual(self.execute_calls, 0)

    async def test_ordinary_approval_without_tool_name_still_works(self):
        waiting = await self.create_run("帮我预约课程")
        self.assertEqual(waiting.status, "WAITING_APPROVAL")
        await self.runtime.execute_run("run-1", {"status": "APPROVED"})
        saved = await self.state()
        self.assertIn("verified-reservation", json.dumps(saved["messages"], ensure_ascii=False))
        self.assertNotIn("unverified-response", json.dumps(saved["messages"], ensure_ascii=False))
        self.assertIn(("POST", "/execute-reservation"), self.calls)
        self.assertEqual(self.execute_calls, 0)


if __name__ == "__main__":
    unittest.main()
