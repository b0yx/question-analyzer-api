import hashlib
import json
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("question_analyzer")

API_KEY = os.getenv("AGENT_ROUTER_API_KEY")
BASE_URL = os.getenv("AGENT_ROUTER_BASE_URL", "https://agentrouter.org/v1")
MODEL_NAME = os.getenv("AGENT_ROUTER_MODEL", "gpt-6.1-sol")
CACHE_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "1000"))
CORS_ORIGINS = [
    origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()
]

app = FastAPI(
    title="Question Clarity Analyzer API",
    description=(
        "Analyzes Arabic multiple-choice questions for clarity, completeness, "
        "and answer validity using rule-based checks and an LLM."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_client: OpenAI | None = None


class AnalyzeRequest(BaseModel):
    question: str = Field(..., min_length=1)
    options: list[str] = Field(..., min_length=2)
    correct_answer: str = Field(..., min_length=1)


def get_client() -> OpenAI:
    """Create the OpenAI-compatible client lazily so the app can start without a key."""
    global _client
    if _client is None:
        if not API_KEY:
            raise RuntimeError("AGENT_ROUTER_API_KEY is not configured.")
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _client


def rule_based_check(data: AnalyzeRequest) -> list[str]:
    errors: list[str] = []
    q = data.question.strip()
    cleaned_opts = [opt.strip() for opt in data.options if opt.strip()]
    ans = data.correct_answer.strip()

    if not q:
        errors.append("نص السؤال فارغ.")
    if "___" in q or "..." in q:
        errors.append("يحتوي نص السؤال على فراغات غير مكتملة.")
    if len(cleaned_opts) < 2:
        errors.append("يجب إدخال خيارين على الأقل.")
    if len(set(cleaned_opts)) != len(cleaned_opts):
        errors.append("توجد خيارات مكررة أو متطابقة.")
    if ans not in cleaned_opts:
        errors.append("الإجابة المعتمدة غير مطابقة لأي خيار من الخيارات المدخلة.")

    return errors


SYSTEM_PROMPT = """
أنت خبير قياس وتقويم تربوي ولغوي متخصص في أسئلة الاختيار من متعدد باللغة العربية.
مهمتك فحص السؤال وخياراته ومفتاح الحل لكشف:
1. الغموض اللغوي أو الدلالي.
2. نقص المعطيات اللازمة للحل.
3. تعدد الإجابات الصحيحة أو تعارض الخيارات.

ثم تقديم صياغة بديلة واضحة تحافظ تمامًا على الهدف التعليمي للسؤال دون تغيير مستواه أو فكرته.
أعد النتيجة بصيغة JSON فقط وفق البنية التالية:
{
  "clarity_score": 0,
  "issues": [
    {
      "type": "ambiguity | missing_info | multiple_correct | conflicting_options",
      "location": "موضع اللبس",
      "explanation": "شرح مختصر ودقيق"
    }
  ],
  "improved_question": "الصياغة البديلة المحكمة",
  "improved_options": ["خيار 1", "خيار 2"],
  "improvement_reason": "سبب التعديل والحفاظ على الهدف التعليمي"
}

أمثلة (Few-shot):
مثال 1 (نقص معطيات):
السؤال: "احسب سرعة الجسم بعد ثانيتين." | الخيارات: ["10 م/ث", "20 م/ث", "30 م/ث"] | الإجابة: "20 م/ث"
النتيجة:
{"clarity_score": 35, "issues": [{"type": "missing_info", "location": "نص السؤال", "explanation": "لم يُذكر التسارع أو السرعة الابتدائية أو نوع الحركة."}], "improved_question": "يتحرك جسم من السكون بتسارع منتظم مقداره 10 م/ث²، ما سرعته بعد ثانيتين؟", "improved_options": ["10 م/ث", "20 م/ث", "30 م/ث"], "improvement_reason": "تمت إضافة المعطيات الفيزيائية اللازمة للحل مع الإبقاء على نفس الخيارات والهدف التعليمي."}

مثال 2 (تعدد إجابات صحيحة):
السؤال: "أي مما يلي من لغات البرمجة؟" | الخيارات: ["Python", "JavaScript", "HTML", "CSS"] | الإجابة: "Python"
النتيجة:
{"clarity_score": 40, "issues": [{"type": "multiple_correct", "location": "الخيارات", "explanation": "كل من Python وJavaScript لغتا برمجة، مما يسبب تضاربًا مع مفتاح الحل."}], "improved_question": "أي مما يلي يُعد لغة برمجة تُستخدم عادةً في تطوير منطق الخوادم والبيانات؟", "improved_options": ["Python", "HTML", "CSS", "JSON"], "improvement_reason": "تم ضبط السؤال ليكون له خيار واحد صحيح لا لبس فيه مع الحفاظ على الهدف التعليمي."}
""".strip()


def make_cache_key(data: AnalyzeRequest) -> str:
    payload = json.dumps(
        {
            "question": data.question.strip(),
            "options": [o.strip() for o in data.options],
            "correct_answer": data.correct_answer.strip(),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_report(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```")
        text = text.removesuffix("```").strip()
    report = json.loads(text or "{}")
    if not isinstance(report, dict):
        raise ValueError("The model did not return a JSON object.")
    return report


def cache_report(key: str, report: dict[str, Any]) -> None:
    CACHE[key] = report
    CACHE.move_to_end(key)
    while len(CACHE) > CACHE_MAX_ENTRIES:
        CACHE.popitem(last=False)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Question Clarity Analyzer API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "api_key_configured": bool(API_KEY),
    }


@app.post("/analyze")
def analyze_question(data: AnalyzeRequest) -> dict[str, Any]:
    rule_errors = rule_based_check(data)
    if rule_errors:
        return {
            "status": "rule_check_failed",
            "errors": rule_errors,
            "input": data.model_dump(),
        }

    key = make_cache_key(data)
    if key in CACHE:
        CACHE.move_to_end(key)
        return {"status": "ok", "cached": True, "report": CACHE[key]}

    user_prompt = (
        f"السؤال: {data.question.strip()}\n"
        f"الخيارات: {json.dumps([o.strip() for o in data.options], ensure_ascii=False)}\n"
        f"الإجابة المعتمدة: {data.correct_answer.strip()}"
    )

    try:
        response = get_client().chat.completions.create(
            model=MODEL_NAME,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        report = parse_report(content)
        cache_report(key, report)
        return {"status": "ok", "cached": False, "report": report}
    except Exception:
        logger.exception("Question analysis failed")
        return {
            "status": "error",
            "message": "تعذر إتمام التحليل حاليًا. يرجى المحاولة لاحقًا.",
            "input": data.model_dump(),
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
