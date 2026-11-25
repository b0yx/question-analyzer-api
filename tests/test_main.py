import json

from fastapi.testclient import TestClient

import main

client = TestClient(main.app)

VALID_REPORT = {
    "clarity_score": 95,
    "issues": [],
    "improved_question": "ما عاصمة فرنسا؟",
    "improved_options": ["باريس", "لندن", "روما"],
    "improvement_reason": "السؤال واضح ولا يحتاج إلى تعديل.",
}

REQUEST = {
    "question": "ما عاصمة فرنسا؟",
    "options": ["باريس", "لندن", "روما"],
    "correct_answer": "باريس",
}


class FakeCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **_: object) -> object:
        self.calls += 1

        class Message:
            content = json.dumps(VALID_REPORT, ensure_ascii=False)

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        return Response()


class FakeChat:
    def __init__(self) -> None:
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self) -> None:
        self.chat = FakeChat()


def setup_function() -> None:
    main.CACHE.clear()


def test_root_returns_metadata() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_health_returns_service_status() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == main.MODEL_NAME
    assert isinstance(body["api_key_configured"], bool)


def test_analyze_rejects_invalid_input_without_calling_model(monkeypatch) -> None:
    fake_client = FakeClient()
    monkeypatch.setattr(main, "get_client", lambda: fake_client)

    response = client.post(
        "/analyze",
        json={
            "question": "ما عاصمة فرنسا؟",
            "options": ["باريس", "لندن"],
            "correct_answer": "روما",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rule_check_failed"
    assert body["errors"]
    assert fake_client.chat.completions.calls == 0


def test_analyze_returns_report_and_uses_cache(monkeypatch) -> None:
    fake_client = FakeClient()
    monkeypatch.setattr(main, "get_client", lambda: fake_client)

    first = client.post("/analyze", json=REQUEST)
    second = client.post("/analyze", json=REQUEST)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["status"] == "ok"
    assert first.json()["cached"] is False
    assert first.json()["report"] == VALID_REPORT
    assert second.json()["cached"] is True
    assert fake_client.chat.completions.calls == 1


def test_analyze_handles_upstream_failure(monkeypatch) -> None:
    def raise_error() -> None:
        raise RuntimeError("upstream unavailable")

    monkeypatch.setattr(main, "get_client", raise_error)

    response = client.post("/analyze", json=REQUEST)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert body["input"] == REQUEST
