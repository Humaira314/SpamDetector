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


@app.get("/health", response_class=JSONResponse)
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


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
    :root {{
      --bg: #f7f3ee;
      --card: #ffffff;
      --text: #222222;
      --muted: #6b6b6b;
      --spam: #ffe2e2;
      --spam-border: #ff6b6b;
      --ham: #e2f6e9;
      --ham-border: #2e9f63;
      --accent: #1f4b99;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    .wrap {{ max-width: 960px; margin: 32px auto; padding: 0 16px; }}
    .header {{ display: flex; flex-direction: column; gap: 8px; margin-bottom: 24px; }}
    .header h1 {{ margin: 0; font-size: 32px; }}
    .header p {{ margin: 0; color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: 1.3fr 1fr; gap: 20px; }}
    .card {{ background: var(--card); padding: 16px; border-radius: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.08); }}
    label {{ font-weight: 600; display: block; margin-bottom: 8px; }}
    textarea {{
      width: 100%;
      min-height: 280px;
      padding: 12px;
      border-radius: 10px;
      border: 1px solid #d6d6d6;
      font-size: 14px;
      resize: vertical;
    }}
    button {{
      margin-top: 12px;
      background: var(--accent);
      color: #ffffff;
      border: none;
      padding: 10px 18px;
      border-radius: 10px;
      font-weight: 600;
      cursor: pointer;
    }}
    .result {{ padding: 16px; border-radius: 12px; border: 2px solid transparent; }}
    .result.spam {{ background: var(--spam); border-color: var(--spam-border); }}
    .result.ham {{ background: var(--ham); border-color: var(--ham-border); }}
    .result.hidden {{ display: none; }}
    .result h2 {{ margin: 0 0 6px 0; }}
    .meta {{ color: var(--muted); font-size: 13px; }}
    .error {{ margin-bottom: 12px; color: #b00020; font-weight: 600; }}
    ul {{ list-style: none; padding: 0; margin: 12px 0 0 0; }}
    li {{ display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #efefef; }}
    li:last-child {{ border-bottom: none; }}
    .term {{ font-weight: 600; }}
    .score {{ color: var(--muted); }}
    .footer {{ margin-top: 16px; color: var(--muted); font-size: 12px; }}
    @media (max-width: 820px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"header\">
      <h1>Spam Detector</h1>
      <p>Paste a full email message to evaluate whether it is spam or ham.</p>
    </div>

    {error_html}

    <div class=\"grid\">
      <form class=\"card\" method=\"post\" action=\"/predict\">
        <label for=\"message\">Email message</label>
        <textarea id=\"message\" name=\"message\" placeholder=\"Paste the email text here...\">{safe_message}</textarea>
        <button type=\"submit\">Analyze</button>
        <div class=\"footer\">Model: {escape(str(DEFAULT_MODEL_PATH))}</div>
        <div class=\"footer\">Threshold: {DEFAULT_THRESHOLD:.2f} | Low-confidence label: {escape(DEFAULT_LOW_CONF_LABEL)}</div>
        <div class=\"footer\">Allowlist: {escape(str(DEFAULT_WHITELIST_PATH))}</div>
      </form>

      <div class=\"card\">
        <div class=\"{label_class}\">
          <h2>{result_text}</h2>
          <div class=\"meta\">Confidence: {confidence_text}</div>
          <div class=\"meta\">{escape(rule_text)}</div>
        </div>
        <div class=\"meta\">Top keywords</div>
        <ul>
          {keywords_items}
        </ul>
      </div>
    </div>
  </div>
</body>
</html>"""
    return HTMLResponse(html)


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
