"""Loopback-only runtime entry point used by the experiment subprocess owner."""

import os

import uvicorn

from zhikexing_agent.api import create_app


if __name__ == "__main__":
    uvicorn.run(
        create_app(),
        host=os.getenv("AGENT_EXPERIMENT_BIND_HOST", "127.0.0.1"),
        port=int(os.environ["PORT"]),
        workers=1,
        access_log=False,
        loop="asyncio:SelectorEventLoop",
        log_level="warning",
    )
