import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
import uvicorn

app = FastAPI(title="客服质检本地审计 Mock", version="1.0.0")
DATA_PATH = Path(__file__).parent / "data" / "mock_audit_records.json"


def get_record(call_id):
    records = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    record = records.get(call_id)
    if record is None:
        raise HTTPException(status_code=404, detail="call not found")
    return record


@app.get("/mock/calls/{call_id}/agent-actions")
def agent_actions(call_id: str):
    return {"actions": get_record(call_id)["actions"]}


@app.get("/mock/calls/{call_id}/crm-summary")
def crm_summary(call_id: str):
    return {"summary": get_record(call_id)["crmSummary"]}


@app.get("/mock/calls/{call_id}/follow-up-tasks")
def follow_up_tasks(call_id: str):
    return {"tasks": get_record(call_id)["followUpTasks"]}


@app.get("/mock/calls/{call_id}/dispute-tickets")
def dispute_tickets(call_id: str):
    return {"tickets": get_record(call_id)["disputeTickets"]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5002)
