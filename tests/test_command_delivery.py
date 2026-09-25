"""Java RocketMQ consumer 的 HTTP 确认边界：数据库提交后才能返回成功。"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from sqlalchemy import func, select

from zhikexing_agent.api import create_app
from zhikexing_agent.config import Settings
from zhikexing_agent.db import Message, Run, RunEvent


class CommandDeliveryTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.directory = tempfile.TemporaryDirectory(prefix="zhikexing-delivery-test-")
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.app = create_app(
            Settings(
                database_url=f"sqlite+aiosqlite:///{root / 'agent.db'}",
                storage_root=root,
                internal_token="isolated-token",
                provider="fixture",
                api_key="",
                max_parallel=0,
            )
        )
        self.lifespan = self.app.router.lifespan_context(self.app)
        await self.lifespan.__aenter__()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://runtime.test"
        )
        self.headers = {
            "X-Internal-Token": "isolated-token",
            "X-Actor-Id": "user-1",
            "X-Workspace-Id": "space-1",
            "X-Command-Delivery": "true",
        }
        response = await self.client.post(
            "/internal/v1/conversations", headers=self.headers, json={"title": "投递测试"}
        )
        self.assertEqual(response.status_code, 201)
        self.body = {
            "runId": "run-1",
            "conversationId": response.json()["id"],
            "input": "hello",
            "mode": "chat",
        }

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.lifespan.__aexit__(None, None, None)

    async def test_redelivery_is_persisted_once_and_conflicts_are_rejected(self):
        for _ in range(2):
            response = await self.client.post(
                "/internal/v1/runs", headers=self.headers, json=self.body
            )
            self.assertEqual(response.status_code, 202)
            async with self.app.state.db.session() as session:
                run = await session.get(Run, "run-1")
                self.assertEqual(run.actor_id, "user-1")
                self.assertEqual(await session.scalar(select(func.count()).select_from(Message)), 1)
        response = await self.client.post(
            "/internal/v1/runs", headers=self.headers, json={**self.body, "input": "different"}
        )
        self.assertEqual(response.status_code, 409)

    async def test_cancel_redelivery_has_one_durable_terminal_event(self):
        await self.client.post("/internal/v1/runs", headers=self.headers, json=self.body)
        for _ in range(2):
            response = await self.client.post(
                "/internal/v1/runs/run-1/cancel", headers=self.headers
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "CANCELLED")
        async with self.app.state.db.session() as session:
            self.assertEqual(await session.scalar(select(func.count()).select_from(RunEvent)), 1)

    async def test_untrusted_identity_cannot_deliver(self):
        response = await self.client.post(
            "/internal/v1/runs",
            headers={**self.headers, "X-Internal-Token": "wrong"},
            json=self.body,
        )
        self.assertEqual(response.status_code, 401)
        response = await self.client.post(
            "/internal/v1/runs", headers={**self.headers, "X-Actor-Id": "other"}, json=self.body
        )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
