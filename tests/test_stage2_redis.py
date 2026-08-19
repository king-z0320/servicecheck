from __future__ import annotations

import os
import time

import pytest


pytestmark = pytest.mark.redis


@pytest.fixture()
def redis_client():
    redis = pytest.importorskip("redis")
    client = redis.Redis.from_url(
        os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
        decode_responses=True,
    )
    try:
        client.ping()
    except Exception as exc:
        pytest.skip(f"Redis integration unavailable: {exc}")
    return client


def test_stream_consumer_group_ack_pending_and_autoclaim(redis_client):
    stream = f"qc:test:stage2:{time.time_ns()}"
    group = "qc-test-workers"
    consumer_a = "worker-a"
    consumer_b = "worker-b"
    redis_client.xgroup_create(stream, group, id="0-0", mkstream=True)
    message_id = redis_client.xadd(
        stream,
        {
            "schema_version": "batch-item-v1",
            "batch_id": "B-REDIS",
            "item_id": "1",
            "idempotency_key": "idem-1",
        },
    )

    first = redis_client.xreadgroup(
        group, consumer_a, {stream: ">"}, count=1, block=100
    )
    assert first[0][1][0][0] == message_id
    pending = redis_client.xpending(stream, group)
    assert pending["pending"] == 1

    claimed = redis_client.xautoclaim(
        stream,
        group,
        consumer_b,
        min_idle_time=0,
        start_id="0-0",
        count=1,
    )
    assert claimed[1][0][0] == message_id
    assert redis_client.xack(stream, group, message_id) == 1
    assert redis_client.xpending(stream, group)["pending"] == 0
    redis_client.delete(stream)
