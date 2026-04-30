from __future__ import annotations

import os
from html import escape
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse

from .cli import predict_with_confidence
from .rules import apply_rules, load_whitelist
from .train import load_model

DEFAULT_MODEL_PATH = Path(os.getenv("SPAMDETECTOR_MODEL", "models/best_model.joblib"))
DEFAULT_THRESHOLD = float(os.getenv("SPAMDETECTOR_THRESHOLD", "0.7"))
DEFAULT_LOW_CONF_LABEL = os.getenv("SPAMDETECTOR_LOW_CONFIDENCE_LABEL", "uncertain")
DEFAULT_WHITELIST_PATH = Path(os.getenv("SPAMDETECTOR_WHITELIST", "config/whitelist.txt"))

app = FastAPI(title="Spam Detector")

_MODEL: Optional[object] = None
_MODEL_ERROR: Optional[str] = None
_WHITELIST: Optional[List[str]] = None


def get_model() -> Optional[object]:
    global _MODEL, _MODEL_ERROR
    if _MODEL is None and _MODEL_ERROR is None:
        if not DEFAULT_MODEL_PATH.exists():
            _MODEL_ERROR = (
                f"Model not found at {DEFAULT_MODEL_PATH}. "
                "Train a model first with: spamdetector train --data data/processed/sms.csv --model both"
            )
            return None
        try:
            _MODEL = load_model(str(DEFAULT_MODEL_PATH))
        except Exception as exc:  # pragma: no cover - runtime error path
            _MODEL_ERROR = f"Failed to load model: {exc}"
            return None
    return _MODEL


def get_model_error() -> Optional[str]:
    return _MODEL_ERROR


def get_whitelist() -> List[str]:
    global _WHITELIST
    if _WHITELIST is None:
        _WHITELIST = list(load_whitelist(str(DEFAULT_WHITELIST_PATH)))
    return _WHITELIST


def top_keywords(model: object, text: str, top_n: int = 8) -> List[Tuple[str, float]]:
    try:
        vectorizer = model.named_steps["tfidf"]
    except Exception:
        return []

    vector = vectorizer.transform([text])
    if vector.nnz == 0:
        return []

    feature_names = vectorizer.get_feature_names_out()
    coo = vector.tocoo()
    scored = list(zip(coo.col, coo.data))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [(feature_names[idx], float(score)) for idx, score in scored[:top_n]]


def render_page(
    message: str = "",
    result_label: Optional[str] = None,
    confidence: Optional[float] = None,
    keywords: Optional[List[Tuple[str, float]]] = None,
    rule: Optional[str] = None,
    error: Optional[str] = None,
) -> HTMLResponse:
    safe_message = escape(message)
    safe_error = escape(error) if error else ""
    label_class = "result" if result_label else "result hidden"
    if result_label == "spam":
        label_class += " spam"
    elif result_label == "ham":
        label_class += " ham"

    confidence_text = "n/a" if confidence is None else f"{confidence:.4f}"
    result_text = "" if result_label is None else result_label.upper()
    rule_text = "" if not rule else f"Reason: {rule.replace('_', ' ')}"

    keywords_items = ""
    if keywords:
        keywords_items = "".join(
            f"<li><span class=\"term\">{escape(term)}</span><span class=\"score\">{score:.4f}</span></li>"
            for term, score in keywords
        )

    error_html = f"<div class=\"error\">{safe_error}</div>" if error else ""

    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Spam Detector</title>
  <style>
    @import url("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap");

    :root {{
      --bg: #f6f2ed;
      --card: #ffffff;
      --ink: #1c1b18;
      --muted: #6d6a66;
      --border: rgba(24, 24, 24, 0.08);
      --shadow: 0 24px 60px rgba(23, 24, 28, 0.12);
      --accent: #2f6fed;
      --accent-deep: #173a8f;
      --accent-soft: rgba(47, 111, 237, 0.16);
      --spam: #ffe6e3;
      --spam-strong: #ff6161;
      --ham: #e7f7ee;
      --ham-strong: #1e9a63;
      --warning: #b00020;
    }}

    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Space Grotesk", "Segoe UI", Arial, sans-serif;
      background: radial-gradient(circle at top right, #eef3ff 0%, #f6f2ed 45%, #f6f2ed 100%);
      color: var(--ink);
      line-height: 1.5;
      min-height: 100vh;
    }}

    .bg-orb {{
      position: fixed;
      width: 320px;
      height: 320px;
      border-radius: 50%;
      background: radial-gradient(circle at 30% 30%, rgba(47, 111, 237, 0.35), rgba(47, 111, 237, 0));
      opacity: 0.7;
      z-index: 0;
    }}
    .orb-one {{ top: -120px; right: -40px; }}
    .orb-two {{ bottom: -160px; left: -60px; background: radial-gradient(circle at 60% 40%, rgba(0, 169, 143, 0.3), rgba(0, 169, 143, 0)); }}

    .page {{ position: relative; padding: 48px 16px 64px; z-index: 1; }}
    .container {{ max-width: 1080px; margin: 0 auto; }}
    .hero {{ display: flex; align-items: center; justify-content: space-between; gap: 24px; flex-wrap: wrap; }}
    .eyebrow {{ text-transform: uppercase; letter-spacing: 0.22em; font-size: 12px; color: var(--muted); font-weight: 600; }}
    .hero h1 {{ margin: 8px 0 6px; font-size: clamp(32px, 5vw, 54px); }}
    .hero p {{ margin: 0; max-width: 520px; color: var(--muted); }}

    .status-pill {{
      background: rgba(255, 255, 255, 0.85);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 12px 18px;
      box-shadow: var(--shadow);
      display: grid;
      gap: 2px;
    }}
    .status-pill span {{ font-size: 12px; color: var(--muted); }}
    .status-pill strong {{ font-size: 14px; }}

    .grid {{ display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 0.8fr); gap: 24px; margin-top: 28px; }}
    .card {{ background: var(--card); padding: 22px; border-radius: 20px; border: 1px solid var(--border); box-shadow: var(--shadow); }}

    label {{ font-weight: 600; display: block; margin-bottom: 10px; color: var(--muted); font-size: 14px; }}
    textarea {{
      width: 100%;
      min-height: 260px;
      padding: 16px;
      border-radius: 16px;
      border: 1px solid transparent;
      background: #f8f8f7;
      box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.08);
      font-size: 15px;
      resize: vertical;
      transition: box-shadow 0.2s ease, border 0.2s ease, background 0.2s ease;
    }}
    textarea:focus {{
      outline: none;
      background: #ffffff;
      border-color: rgba(47, 111, 237, 0.4);
      box-shadow: 0 0 0 4px var(--accent-soft);
    }}
    textarea::placeholder {{ color: rgba(109, 106, 102, 0.7); }}

    .actions {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
    button {{
      background: linear-gradient(135deg, var(--accent), var(--accent-deep));
      color: #ffffff;
      border: none;
      padding: 12px 22px;
      border-radius: 999px;
      font-weight: 600;
      cursor: pointer;
      box-shadow: 0 14px 28px rgba(47, 111, 237, 0.24);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    button:hover {{ transform: translateY(-1px); box-shadow: 0 18px 32px rgba(47, 111, 237, 0.28); }}

    .hint {{ font-size: 12px; color: var(--muted); }}
    .meta-stack {{ margin-top: 16px; display: grid; gap: 6px; font-size: 12px; color: var(--muted); }}

    .result {{ padding: 18px; border-radius: 18px; border: 2px solid transparent; background: #f7f6f4; min-height: 130px; animation: floatIn 0.45s ease; }}
    .result.spam {{ background: var(--spam); border-color: var(--spam-strong); }}
    .result.ham {{ background: var(--ham); border-color: var(--ham-strong); }}
    .result.hidden {{ display: none; }}
    .result h2 {{ margin: 4px 0 6px; font-size: 26px; letter-spacing: 0.02em; }}
    .result-label {{ text-transform: uppercase; font-size: 11px; letter-spacing: 0.2em; color: var(--muted); }}
    .meta {{ color: var(--muted); font-size: 13px; }}
    .headline {{ margin-top: 16px; font-weight: 600; color: var(--muted); }}

    .keywords {{ list-style: none; padding: 0; margin: 12px 0 0 0; display: grid; gap: 8px; }}
    .keywords li {{ display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; border-radius: 12px; border: 1px solid var(--border); background: #ffffff; }}
    .term {{ font-weight: 600; }}
    .score {{ color: var(--muted); font-variant-numeric: tabular-nums; }}

    .error {{ margin: 18px 0; padding: 12px 16px; background: #ffecec; border: 1px solid #ffc3c3; border-radius: 12px; color: var(--warning); font-weight: 600; }}

    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .status-pill {{ width: 100%; }}
    }}

    @keyframes floatIn {{
      from {{ transform: translateY(6px); opacity: 0; }}
      to {{ transform: translateY(0); opacity: 1; }}
    }}
  </style>
</head>
<body>
  <div class=\"bg-orb orb-one\"></div>
  <div class=\"bg-orb orb-two\"></div>
  <div class=\"page\">
    <div class=\"container\">
      <header class=\"hero\">
        <div>
          <div class=\"eyebrow\">AI message classifier</div>
          <h1>Spam Detector</h1>
          <p>Paste a full email or SMS message. Get a prediction with confidence and top terms.</p>
        </div>
        <div class=\"status-pill\">
          <span>Service status</span>
          <strong>Ready for checks</strong>
        </div>
      </header>

      {error_html}

      <div class=\"grid\">
        <form class=\"card\" method=\"post\" action=\"/predict\">
          <label for=\"message\">Message</label>
          <textarea id=\"message\" name=\"message\" placeholder=\"Paste the email or SMS text here...\">{safe_message}</textarea>
          <div class=\"actions\">
            <button type=\"submit\">Analyze</button>
            <span class=\"hint\">Tip: include subject lines and signatures for best results.</span>
          </div>
          <div class=\"meta-stack\">
            <div>Model: {escape(str(DEFAULT_MODEL_PATH))}</div>
            <div>Threshold: {DEFAULT_THRESHOLD:.2f} | Low-confidence label: {escape(DEFAULT_LOW_CONF_LABEL)}</div>
            <div>Allowlist: {escape(str(DEFAULT_WHITELIST_PATH))}</div>
          </div>
        </form>

        <div class=\"card\">
          <div class=\"{label_class}\">
            <div class=\"result-label\">Prediction</div>
            <h2>{result_text}</h2>
            <div class=\"meta\">Confidence: {confidence_text}</div>
            <div class=\"meta\">{escape(rule_text)}</div>
          </div>
          <div class=\"headline\">Top keywords</div>
          <ul class=\"keywords\">
            {keywords_items}
          </ul>
        </div>
      </div>
    </div>
  </div>
</body>
</html>"""
    return HTMLResponse(html)


@app.get("/health", response_class=JSONResponse)
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    model = get_model()
    error = get_model_error() if model is None else None
    return render_page(error=error)


@app.post("/predict", response_class=HTMLResponse)
def predict(message: str = Form("")) -> HTMLResponse:
    model = get_model()
    error = get_model_error() if model is None else None
    if error:
        return render_page(message=message, error=error)

    if not message.strip():
        return render_page(message=message, error="Please paste a message to analyze.")

    result = predict_with_confidence(model, [message])[0]
    label, confidence, rule = apply_rules(
        label=result["label"],
        confidence=result["confidence"],
        text=message,
        whitelist=get_whitelist(),
        threshold=DEFAULT_THRESHOLD,
        low_confidence_label=DEFAULT_LOW_CONF_LABEL,
    )
    keywords = top_keywords(model, message)
    return render_page(
        message=message,
        result_label=label,
        confidence=confidence,
        keywords=keywords,
        rule=rule,
    )
