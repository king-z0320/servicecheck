from __future__ import annotations

from sqlalchemy import select

from qc.database import create_session_factory, database_url_from_env
from qc.models import QualityReport
from qc.orm_models import QCReportRow, QCRunRow, ReviewTaskRow
from qc.review_service import compute_route_reasons, needs_review_task
from qc.review_store import ensure_review_task_in_session


def backfill_review_tasks(database_url: str | None = None) -> int:
    url = database_url or database_url_from_env()
    engine = create_database_engine_safe(url)
    factory = create_session_factory(engine)
    created = 0
    try:
        with factory.begin() as session:
            rows = session.execute(
                select(QCRunRow, QCReportRow)
                .join(QCReportRow, QCReportRow.run_id == QCRunRow.run_id)
                .where(QCRunRow.status == "PARTIAL")
            ).all()
            for run, report_row in rows:
                report = QualityReport.model_validate(report_row.report_json)
                if not needs_review_task(run.status, report):
                    continue
                existing = session.scalar(
                    select(ReviewTaskRow.review_task_id).where(
                        ReviewTaskRow.run_id == run.run_id
                    )
                )
                if existing is not None:
                    continue
                reasons = compute_route_reasons(run.status, report, [])
                ensure_review_task_in_session(session, run.run_id, reasons)
                created += 1
    finally:
        engine.dispose()
    return created


def create_database_engine_safe(database_url: str):
    from qc.database import create_database_engine

    return create_database_engine(database_url)


def main() -> None:
    count = backfill_review_tasks()
    print(f"backfill ensured {count} review task rows")


if __name__ == "__main__":
    main()
