"""单 worker 入口：持久化调度器与内部 API 共用生命周期。"""

import logging
import os

import uvicorn

from .api import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
# Psycopg 异步连接在 Windows 需要 Selector 事件循环，Linux 同样兼容。
uvicorn.run(
    create_app(),
    host="0.0.0.0",
    port=int(os.getenv("PORT", "8000")),
    workers=1,
    access_log=False,
    loop="asyncio:SelectorEventLoop",
)
