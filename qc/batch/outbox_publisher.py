from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from opentelemetry import propagate

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OutboxEvent:
    event_id: str
    event_type: str
    batch_id: str
    item_id: int
    idempotency_key: str
    created_at: datetime
    attempts: int = 0

    def payload(self) -> dict[str, Any]:
        carrier = {
            "schema_version": "batch-item-v1",
            "event_id": self.event_id,
            "event_type": self.event_type,
            "batch_id": self.batch_id,
            "item_id": self.item_id,
            "idempotency_key": self.idempotency_key,
        }
        propagate.inject(carrier)
        return carrier


class OutboxPublisher:
    """Publish durable database events to Redis; duplicates are acceptable."""

    def __init__(
        self,
        store,
        redis_client,
        *,
        stream: str = "qc:batch-items:v1",
        max_events: int = 50,
        max_attempts: int = 5,
        backoff_initial: float = 1.0,
        backoff_max: float = 30.0,
    ):
        self.store = store
        self.redis = redis_client
        self.stream = stream
        self.max_events = max_events
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_initial = max(0.0, float(backoff_initial))
        self.backoff_max = max(0.0, float(backoff_max))

    def _retry_delay(self, attempts: int) -> float:
        return min(
            self.backoff_initial * (2 ** max(0, attempts)),
            self.backoff_max,
        )

    def publish_once(self) -> int:
        events = self.store.pending_outbox_events(limit=self.max_events)
        published = 0
        for event in events:
            try:
                message_id = self.redis.xadd(self.stream, event.payload())
                self.store.mark_outbox_published(event.event_id, str(message_id))
                published += 1
            except Exception as exc:
                error_summary = type(exc).__name__
                self.store.mark_outbox_failed(
                    event.event_id,
                    error_summary,
                    max_attempts=self.max_attempts,
                    retry_delay_seconds=self._retry_delay(event.attempts),
                )
                LOGGER.warning(
                    "outbox publish failed",
                    extra={"event_id": event.event_id, "error_code": error_summary},
                )
        return published

    def run_forever(self, poll_seconds: float = 1.0):
        while True:
            self.publish_once()
            time.sleep(max(0.01, poll_seconds))


def main() -> int:
    from qc.batch.postgres_store import PostgresBatchStore
    from qc.batch.worker import build_redis_client
    from qc.database import database_url_from_env
    from qc.observability.runtime import configure_local_observability
    from pathlib import Path

    configure_local_observability(Path(__file__).resolve().parents[2], process_name="publisher")

    publisher = OutboxPublisher(
        PostgresBatchStore(database_url_from_env()),
        build_redis_client(),
    )
    publisher.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
