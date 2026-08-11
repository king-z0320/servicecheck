from qc.event_extractor import EventExtractor
from qc.models import EventType, TranscriptTurn


class FakeGateway:
    def complete_json(self, system, user, schema):
        return {
            "events": [
                {
                    "eventId": "E001",
                    "type": "REPAYMENT_DISPUTE",
                    "statement": "我已经还完了",
                    "turnIds": ["T0001"],
                    "confidence": 0.98,
                    "ambiguous": False,
                }
            ]
        }


def test_extracts_customer_claim_with_real_turn_reference():
    extractor = EventExtractor(FakeGateway())
    events = extractor.extract([
        TranscriptTurn(
            turnId="T0001",
            speaker="客户",
            text="我已经还完了",
            start=1,
            end=2,
        )
    ])
    assert events[0].type == EventType.REPAYMENT_DISPUTE
    assert events[0].turnIds == ["T0001"]


def test_rejects_hallucinated_turn_ids():
    class BadGateway(FakeGateway):
        def complete_json(self, system, user, schema):
            data = super().complete_json(system, user, schema)
            data["events"][0]["turnIds"] = ["T9999"]
            return data

    extractor = EventExtractor(BadGateway())
    events = extractor.extract([
        TranscriptTurn(
            turnId="T0001",
            speaker="客户",
            text="我已经还完了",
            start=1,
            end=2,
        )
    ])
    assert events == []
#测试的是 EventExtractor 类的 extract 方法，确保它不会接受不存在的 turnId。
#测试方法：创建一个模拟网关，返回一个包含不存在的 turnId 的事件，然后调用 extract 方法，期望返回空列表。
#assert的意思是：断言，如果条件不成立，则抛出 AssertionError 异常。