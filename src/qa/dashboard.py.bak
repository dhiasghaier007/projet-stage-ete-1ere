"""
RAG Performance Dashboard — Stage 5 QA deliverable.

Reads whatever evaluation artifacts already exist on disk:
  - data/outputs/quality_reports/*.json      (from quality_report.py)
  - data/outputs/classification_eval_report.json  (from eval_classifiers.py)
  - data/outputs/gold_score_report.json      (from score_against_gold.py)

...and renders them into a single static HTML file. Deliberately NOT a live
web server — this is meant to be regenerated after each ingestion/eval run
and opened locally or attached to a report, matching how the rest of this
pipeline is used (scripts you run, not services you keep up).

Consistent with the rest of the pipeline: an artifact that hasn't been
generated yet is shown as an explicit "not yet run" state, never silently
skipped or faked with placeholder numbers.
"""
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
_OUTPUTS_DIR = REPO_ROOT / "data" / "outputs"
_QUALITY_REPORTS_DIR = _OUTPUTS_DIR / "quality_reports"
_CLASSIFICATION_REPORT_PATH = _OUTPUTS_DIR / "classification_eval_report.json"
_GOLD_REPORT_PATH = _OUTPUTS_DIR / "gold_score_report.json"
_DASHBOARD_OUTPUT_PATH = _OUTPUTS_DIR / "dashboard.html"


def _load_quality_reports(reports_dir: Path = _QUALITY_REPORTS_DIR) -> List[Dict[str, Any]]:
    if not reports_dir.exists():
        return []
    reports = []
    for path in sorted(reports_dir.glob("quality_report_*.json")):
        try:
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue  # skip a corrupt/partial file rather than crash the whole dashboard
    return reports


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _esc(value: Any) -> str:
    """HTML-escape anything we're about to print, since report content
    (filenames, mistake values) ultimately traces back to document content
    and should never be trusted to be safe HTML."""
    return html.escape(str(value))


def _score_color(score: Optional[float]) -> str:
    if score is None:
        return "var(--muted)"
    if score >= 85:
        return "var(--good)"
    if score >= 65:
        return "var(--warn)"
    return "var(--bad)"


def _render_trend_svg(reports: List[Dict[str, Any]]) -> str:
    """Builds a small SVG line chart of overall_score over time by hand —
    no charting library dependency. Returns an empty-state message if fewer
    than 2 points exist, since a single point can't show a trend."""
    points = [(r.get("generated_at", ""), r.get("overall_score")) for r in reports if r.get("overall_score") is not None]
    if len(points) < 2:
        return '<p class="muted">Not enough history yet for a trend line — run quality_report.py again after your next ingestion to start building one.</p>'

    width, height, pad = 640, 160, 24
    scores = [s for _, s in points]
    min_score, max_score = min(scores) - 5, max(scores) + 5
    span = max(max_score - min_score, 1)
    step_x = (width - 2 * pad) / (len(points) - 1)

    def y_for(score: float) -> float:
        return height - pad - ((score - min_score) / span) * (height - 2 * pad)

    coords = [(pad + i * step_x, y_for(score)) for i, (_, score) in enumerate(points)]
    path_d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(coords))
    circles = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{_score_color(score)}">'
        f'<title>{_esc(ts)}: {score}</title></circle>'
        for (ts, score), (x, y) in zip(points, coords)
    )

    return f"""
    <svg viewBox="0 0 {width} {height}" class="trend-svg" role="img" aria-label="Corpus quality score over time">
      <path d="{path_d}" fill="none" stroke="var(--accent)" stroke-width="2" />
      {circles}
    </svg>
    """


def _render_metrics_table(metrics: Dict[str, Dict[str, Any]]) -> str:
    rows = []
    for name, metric in metrics.items():
        score = metric.get("score")
        score_str = f"{score:.1f}" if score is not None else "N/A"
        color = _score_color(score)
        label = name.replace("_", " ").title()
        rows.append(f"""
        <tr>
          <td>{_esc(label)}</td>
          <td><span class="badge" style="background:{color}">{score_str}</span></td>
          <td class="detail">{_esc(metric.get('detail', ''))}</td>
        </tr>""")
    return "\n".join(rows)


def _render_drift_banner(drift: Dict[str, Any]) -> str:
    status = drift.get("status")
    if status == "no_baseline":
        return '<div class="banner muted">No prior report to compare against yet — this was the first run.</div>'
    if status == "stable":
        compared = _esc(drift.get("compared_against", "previous run"))
        return f'<div class="banner good">✅ Stable — no metric regressed vs. {compared}.</div>'
    if status == "drift":
        compared = _esc(drift.get("compared_against", "previous run"))
        items = "".join(
            f"<li>{_esc(r['metric'].replace('_', ' ').title())}: "
            f"{r['previous_score']} → {r['current_score']} ({r['delta']:+.1f} points)</li>"
            for r in drift.get("regressions", [])
        )
        return f'<div class="banner bad">⚠️ Drift detected vs. {compared}:<ul>{items}</ul></div>'
    return ""


def _render_classification_section(report: Optional[Dict[str, Any]]) -> str:
    if report is None:
        return ('<p class="muted">Not yet run. Generate this with: '
                '<code>python3 src/classification/eval_classifiers.py</code></p>')
    overall = report.get("overall_accuracy_pct")
    overall_str = f"{overall:.1f}%" if overall is not None else "N/A"
    field_rows = ""
    for field, stats in report.get("field_accuracy", {}).items():
        total = stats.get("total", 0)
        correct = stats.get("correct", 0)
        pct = f"{100 * correct / total:.1f}%" if total else "N/A"
        field_rows += f"<tr><td>{_esc(field)}</td><td>{pct}</td><td>{correct}/{total}</td></tr>"
    return f"""
    <p>Overall exact-match accuracy: <span class="badge" style="background:{_score_color(overall)}">{overall_str}</span>
    ({report.get('correct', 0)}/{report.get('total', 0)} documents fully correct)</p>
    <table><thead><tr><th>Field</th><th>Accuracy</th><th>Correct/Total</th></tr></thead>
    <tbody>{field_rows}</tbody></table>
    <p class="muted">Generated at {_esc(report.get('generated_at', 'unknown'))}</p>
    """


def _render_gold_section(report: Optional[Dict[str, Any]]) -> str:
    if report is None:
        return ('<p class="muted">Not yet run. Generate this with: '
                '<code>python3 src/classification/score_against_gold.py --classified &lt;dir&gt; --labels &lt;gold_labels.jsonl&gt;</code></p>')
    overall = report.get("overall_accuracy_pct")
    overall_str = f"{overall:.1f}%" if overall is not None else "N/A"
    field_rows = ""
    for field, stats in report.get("per_field_accuracy", {}).items():
        acc = stats.get("accuracy_pct")
        acc_str = f"{acc:.1f}%" if acc is not None else "N/A"
        field_rows += f"<tr><td>{_esc(field)}</td><td>{acc_str}</td><td>{stats.get('correct', 0)}/{stats.get('total', 0)}</td></tr>"
    return f"""
    <p>Overall field accuracy vs. gold labels: <span class="badge" style="background:{_score_color(overall)}">{overall_str}</span>
    ({report.get('total_scored', 0)} documents scored, {report.get('skipped_unclassified', 0)} skipped)</p>
    <table><thead><tr><th>Field</th><th>Accuracy</th><th>Correct/Total</th></tr></thead>
    <tbody>{field_rows}</tbody></table>
    <p class="muted">Generated at {_esc(report.get('generated_at', 'unknown'))}</p>
    """


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RAG Pipeline — Performance Dashboard</title>
<style>
  :root {{
    --bg: #0f1115; --card: #171a21; --text: #e6e8ec; --muted: #8b909c;
    --accent: #5b8def; --good: #2f9e5c; --warn: #d99a2b; --bad: #d9463f;
    --border: #262a33;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:32px; background:var(--bg); color:var(--text);
          font-family:-apple-system,Segoe UI,Roboto,sans-serif; }}
  h1 {{ font-size:22px; margin-bottom:4px; }}
  .subtitle {{ color:var(--muted); margin-top:0; margin-bottom:28px; font-size:14px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:10px;
           padding:20px; margin-bottom:20px; }}
  .card h2 {{ font-size:15px; margin-top:0; margin-bottom:14px; color:var(--text); }}
  .score-hero {{ font-size:48px; font-weight:700; }}
  .score-hero.good {{ color:var(--good); }}
  .score-hero.warn {{ color:var(--warn); }}
  .score-hero.bad {{ color:var(--bad); }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--border); }}
  th {{ color:var(--muted); font-weight:500; }}
  td.detail {{ color:var(--muted); }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:12px; color:#0f1115;
            font-weight:600; font-size:12px; }}
  .banner {{ padding:12px 14px; border-radius:8px; margin-bottom:14px; font-size:13px; }}
  .banner.good {{ background:rgba(47,158,92,0.15); border:1px solid var(--good); }}
  .banner.bad {{ background:rgba(217,70,63,0.15); border:1px solid var(--bad); }}
  .banner.muted {{ background:rgba(139,144,156,0.1); border:1px solid var(--border); color:var(--muted); }}
  .banner ul {{ margin:8px 0 0 18px; }}
  .muted {{ color:var(--muted); font-size:13px; }}
  code {{ background:#20242d; padding:1px 5px; border-radius:4px; font-size:12px; }}
  .trend-svg {{ width:100%; height:auto; }}
  .full-width {{ grid-column: 1 / -1; }}
</style>
</head>
<body>
  <h1>RAG Pipeline — Performance Dashboard</h1>
  <p class="subtitle">Generated {generated_at} — regenerate with <code>python3 src/qa/dashboard.py</code> after each ingestion/eval run</p>

  <div class="grid">
    <div class="card">
      <h2>Corpus Quality Score</h2>
      <div class="score-hero {hero_class}">{hero_score}</div>
      <p class="muted">as of {latest_report_time}</p>
    </div>
    <div class="card">
      <h2>Drift Status</h2>
      {drift_banner}
    </div>

    <div class="card full-width">
      <h2>Score Trend</h2>
      {trend_svg}
    </div>

    <div class="card full-width">
      <h2>Latest Corpus Quality Breakdown</h2>
      <table>
        <thead><tr><th>Metric</th><th>Score</th><th>Detail</th></tr></thead>
        <tbody>{metrics_rows}</tbody>
      </table>
    </div>

    <div class="card">
      <h2>Classification Accuracy (labeled test set)</h2>
      {classification_section}
    </div>

    <div class="card">
      <h2>Classification Accuracy (real gold-labeled corpus)</h2>
      {gold_section}
    </div>
  </div>
</body>
</html>
"""


def render_dashboard() -> str:
    reports = _load_quality_reports()
    classification_report = _load_json_if_exists(_CLASSIFICATION_REPORT_PATH)
    gold_report = _load_json_if_exists(_GOLD_REPORT_PATH)

    if reports:
        latest = reports[-1]
        hero_score = latest.get("overall_score")
        hero_str = f"{hero_score:.1f}" if hero_score is not None else "N/A"
        hero_class = "good" if (hero_score or 0) >= 85 else "warn" if (hero_score or 0) >= 65 else "bad"
        drift_banner = _render_drift_banner(latest.get("drift", {"status": "no_baseline"}))
        metrics_rows = _render_metrics_table(latest.get("metrics", {}))
        latest_time = _esc(latest.get("generated_at", "unknown"))
    else:
        hero_str, hero_class = "N/A", "warn"
        drift_banner = ('<div class="banner muted">No quality report found yet. Generate one with: '
                         '<code>python3 -m src.qa.quality_report</code></div>')
        metrics_rows = '<tr><td colspan="3" class="muted">No data yet.</td></tr>'
        latest_time = "never"

    return _TEMPLATE.format(
        generated_at=datetime.now(timezone.utc).isoformat(),
        hero_class=hero_class,
        hero_score=hero_str,
        latest_report_time=latest_time,
        drift_banner=drift_banner,
        trend_svg=_render_trend_svg(reports),
        metrics_rows=metrics_rows,
        classification_section=_render_classification_section(classification_report),
        gold_section=_render_gold_section(gold_report),
    )


def main():
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    html_content = render_dashboard()
    _DASHBOARD_OUTPUT_PATH.write_text(html_content, encoding="utf-8")
    print(f"✅ Dashboard written to {_DASHBOARD_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
