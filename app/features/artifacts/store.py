from __future__ import annotations

import os
from pathlib import Path


def enabled() -> bool:
    return bool(os.getenv("S3_ENDPOINT_URL", "").strip())


def _client():
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("缺少对象存储驱动，请使用一键容器部署") from exc
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name=os.getenv("S3_REGION", "us-east-1"),
    )


def _bucket() -> str:
    return os.getenv("S3_BUCKET", "football-insight")


def ensure_bucket() -> None:
    if not enabled():
        return
    client, bucket = _client(), _bucket()
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        client.create_bucket(Bucket=bucket)


def upload_project_tree(project_id: str, root: Path) -> None:
    if not enabled() or not root.is_dir():
        return
    ensure_bucket()
    client, bucket = _client(), _bucket()
    for path in root.rglob("*"):
        if path.is_file() and not path.name.endswith((".tmp", ".uploading")):
            client.upload_file(str(path), bucket, f"projects/{project_id}/{path.relative_to(root).as_posix()}")


def hydrate_project_tree(project_id: str, root: Path) -> None:
    if not enabled() or root.exists():
        return
    ensure_bucket()
    client, bucket, prefix = _client(), _bucket(), f"projects/{project_id}/"
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            relative = item["Key"][len(prefix):]
            if not relative:
                continue
            destination = root / Path(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, item["Key"], str(destination))


def delete_project_tree(project_id: str) -> None:
    if not enabled():
        return
    client, bucket, prefix = _client(), _bucket(), f"projects/{project_id}/"
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
        if objects:
            client.delete_objects(Bucket=bucket, Delete={"Objects": objects, "Quiet": True})
