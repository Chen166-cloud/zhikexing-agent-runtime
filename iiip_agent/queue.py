"""可靠命令投递：收到命令后先通过内部接口持久化，再确认消息。"""

import asyncio
import json
import logging
import os
import re

import aio_pika
import httpx

logger = logging.getLogger(__name__)


class CommandConsumer:
    def __init__(self, settings):
        self.settings = settings
        self.connection = None
        self.client = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{os.getenv('PORT', '8000')}", timeout=30
        )

    async def start(self):
        self.connection = await aio_pika.connect_robust(os.environ["RABBITMQ_URL"])
        channel = await self.connection.channel()
        await channel.set_qos(prefetch_count=1)
        exchange = await channel.declare_exchange(
            "iiip.agent", aio_pika.ExchangeType.DIRECT, durable=True
        )
        dead_exchange = await channel.declare_exchange(
            "iiip.agent.dlx", aio_pika.ExchangeType.DIRECT, durable=True
        )
        failed = await channel.declare_queue("iiip.agent.failed", durable=True)
        await failed.bind(dead_exchange, routing_key="commands.failed")
        queue = await channel.declare_queue(
            "iiip.agent.commands",
            durable=True,
            arguments={
                "x-dead-letter-exchange": "iiip.agent.dlx",
                "x-dead-letter-routing-key": "commands.failed",
            },
        )
        await queue.bind(exchange, routing_key="commands")
        await queue.consume(self.receive)

    async def receive(self, message):
        try:
            body = json.loads(message.body)
            path = body["path"]
            if path != "/runs" and not re.fullmatch(r"/runs/[a-zA-Z0-9-]{1,64}/cancel", path):
                raise ValueError("命令路径未注册")
            headers = {
                "X-Internal-Token": self.settings.internal_token,
                "X-Actor-Id": str(body["actorId"]),
                "X-Workspace-Id": str(body["workspaceId"]),
                "X-Command-Delivery": "true",
            }
            response = await self.client.post(
                "/internal/v1" + path, headers=headers, json=body["payload"]
            )
            if response.is_success:
                await message.ack()
            elif (
                response.status_code >= 500
                or response.status_code == 404
                and path.endswith("/cancel")
            ):
                # 服务尚未启动或取消先于创建到达时，保留消息，避免丢失取消命令。
                await asyncio.sleep(2)
                await message.nack(requeue=True)
            else:
                logger.warning(
                    "命令进入死信队列 id=%s status=%s", body.get("id"), response.status_code
                )
                await message.reject(requeue=False)
        except (httpx.HTTPError, aio_pika.exceptions.AMQPException):
            await asyncio.sleep(2)
            if not message.processed:
                await message.nack(requeue=True)
        except (ValueError, KeyError, TypeError):
            logger.warning("命令格式无效，转入死信队列")
            await message.reject(requeue=False)

    async def close(self):
        if self.connection:
            await self.connection.close()
        await self.client.aclose()
