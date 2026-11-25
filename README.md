# Question Clarity Analyzer API

FastAPI service that evaluates Arabic multiple-choice questions for clarity, completeness, and answer validity. It combines deterministic validation rules with an LLM-powered review that returns structured findings and an improved version of the question.

## Features

- Rule-based validation for empty inputs, placeholders, duplicate options, and mismatched answers.
- LLM review that detects ambiguity, missing information, multiple correct answers, and conflicting options.
- Structured JSON responses with a clarity score, issues, and an improved question.
- In-memory caching to avoid repeated model calls for identical input.
- Automatic OpenAPI documentation at `/docs`.
- Configurable CORS origins and model settings through environment variables.

## Tech stack

- [FastAPI](https://fastapi.tiangolo.com/)
- [Pydantic](https://docs.pydantic.dev/)
- [OpenAI Python SDK](https://github.com/openai/openai-python) (OpenAI-compatible endpoint)
- [uv](https://docs.astral.sh/uv/) for dependency management
- [pytest](https://docs.pytest.org/) and [Ruff](https://docs.astral.sh/ruff/) for testing and linting

## Getting started

### Prerequisites

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- An API key for an OpenAI-compatible endpoint (AgentRouter by default)

### Installation

```bash
git clone https://github.com/b0yx/question-analyzer-api.git
cd question-analyzer-api
uv sync
cp .env.example .env
```

Edit `.env` and set `AGENT_ROUTER_API_KEY`. Never commit the `.env` file.

### Run the API

```bash
uv run uvicorn main:app --reload
```

Alternatively:

```bash
uv run python main.py
```

The server starts at `http://127.0.0.1:8000`. Interactive docs are available at `http://127.0.0.1:8000/docs`.

### Tests and linting

```bash
uv run pytest
uv run ruff check .
```

## Configuration

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `AGENT_ROUTER_API_KEY` | Yes | — | API key for the LLM gateway. |
| `AGENT_ROUTER_BASE_URL` | No | `https://agentrouter.org/v1` | OpenAI-compatible base URL. |
| `AGENT_ROUTER_MODEL` | No | `gpt-6.1-sol` | Model used for the analysis. |
| `CORS_ORIGINS` | No | `*` | Comma-separated list of allowed origins. |
| `CACHE_MAX_ENTRIES` | No | `1000` | Maximum number of analyses kept in the in-memory cache. |
| `HOST` | No | `0.0.0.0` | Bind host used by `python main.py`. |
| `PORT` | No | `8000` | Bind port used by `python main.py`. |

## API reference

### `GET /`

Returns basic service metadata.

### `GET /health`

Returns service status, the configured model, and whether an API key is present.

```json
{
  "status": "ok",
  "model": "gpt-6.1-sol",
  "api_key_configured": true
}
```

### `POST /analyze`

Analyzes a multiple-choice question.

**Request body**

| Field | Type | Description |
| --- | --- | --- |
| `question` | string | The question text. Must not be empty. |
| `options` | list[string] | At least two answer options. |
| `correct_answer` | string | The approved answer. Must match one of the options. |

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "question": "ما عاصمة فرنسا؟",
    "options": ["باريس", "لندن", "روما"],
    "correct_answer": "باريس"
  }'
```

**Successful response**

```json
{
  "status": "ok",
  "cached": false,
  "report": {
    "clarity_score": 95,
    "issues": [],
    "improved_question": "ما عاصمة فرنسا؟",
    "improved_options": ["باريس", "لندن", "روما"],
    "improvement_reason": "السؤال واضح ولا يحتاج إلى تعديل."
  }
}
```

**Rule validation failure**

Returned when the input breaks a deterministic rule. No model call is made.

```json
{
  "status": "rule_check_failed",
  "errors": ["الإجابة المعتمدة غير مطابقة لأي خيار من الخيارات المدخلة."],
  "input": {
    "question": "ما عاصمة فرنسا؟",
    "options": ["باريس", "لندن"],
    "correct_answer": "روما"
  }
}
```

**Upstream failure**

```json
{
  "status": "error",
  "message": "تعذر إتمام التحليل حاليًا. يرجى المحاولة لاحقًا.",
  "input": {
    "question": "ما عاصمة فرنسا؟",
    "options": ["باريس", "لندن"],
    "correct_answer": "باريس"
  }
}
```

## Project structure

```text
.
├── main.py              # FastAPI app, validation rules, and LLM integration
├── tests/
│   └── test_main.py     # API tests with a mocked LLM client
├── .env.example         # Documented environment variables
├── pyproject.toml       # Project metadata and tool configuration
└── .github/workflows/ci.yml
```

## Notes

- The cache is stored in memory and is cleared when the process restarts. It is not shared between workers.
- The API requires an OpenAI-compatible endpoint that supports JSON response mode.
- Rotate any API key that has been committed or shared. `.env` is excluded from version control.

## License

This project is provided for internal use. Add a license file before distributing it publicly.
