"""本地与 S3 对象存储共用接口；键由服务端生成。"""

import asyncio
import os

import boto3

from .config import Settings


class ObjectStorage:
    def __init__(self, settings: Settings):
        self.root = settings.storage_root / "objects"
        self.root.mkdir(parents=True, exist_ok=True)
        self.bucket = settings.s3_bucket
        self.s3 = None
        if settings.s3_endpoint:
            self.s3 = boto3.client(
                "s3",
                endpoint_url=settings.s3_endpoint,
                aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
                region_name="us-east-1",
            )

    async def put(self, key: str, content: bytes):
        if self.s3:
            await asyncio.to_thread(self.s3.put_object, Bucket=self.bucket, Key=key, Body=content)
        else:
            path = self.root / key
            path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(path.write_bytes, content)

    async def get(self, key: str) -> bytes:
        if self.s3:

            def read():
                result = self.s3.get_object(Bucket=self.bucket, Key=key)
                with result["Body"] as body:
                    return body.read()

            return await asyncio.to_thread(read)
        return await asyncio.to_thread((self.root / key).read_bytes)

    async def delete(self, key: str):
        if self.s3:
            await asyncio.to_thread(self.s3.delete_object, Bucket=self.bucket, Key=key)
        else:
            await asyncio.to_thread((self.root / key).unlink, missing_ok=True)
