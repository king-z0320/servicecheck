import json
from pathlib import Path

from flask import Flask, abort, jsonify

app = Flask(__name__)
DATA_PATH = Path(__file__).parent / "data" / "mock_audit_records.json"


def get_record(call_id):
    records = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    record = records.get(call_id)
    if record is None:
        abort(404)
    return record


@app.get("/mock/calls/<call_id>/agent-actions")
def agent_actions(call_id):
    return jsonify({"actions": get_record(call_id)["actions"]})


@app.get("/mock/calls/<call_id>/crm-summary")
def crm_summary(call_id):
    return jsonify({"summary": get_record(call_id)["crmSummary"]})


@app.get("/mock/calls/<call_id>/follow-up-tasks")
def follow_up_tasks(call_id):
    return jsonify({"tasks": get_record(call_id)["followUpTasks"]})


@app.get("/mock/calls/<call_id>/dispute-tickets")
def dispute_tickets(call_id):
    return jsonify({"tickets": get_record(call_id)["disputeTickets"]})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=False)
