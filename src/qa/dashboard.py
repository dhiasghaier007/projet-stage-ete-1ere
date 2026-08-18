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


def _score_class(score: Optional[float]) -> str:
    if score is None:
        return "muted"
    if score >= 85:
        return "good"
    if score >= 65:
        return "warn"
    return "bad"


def _fmt_time(ts: str) -> str:
    """Best-effort human-friendly timestamp; falls back to the raw string."""
    if not ts or ts == "unknown":
        return "unknown"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y \u00b7 %H:%M UTC")
    except ValueError:
        return ts


def _render_trend_svg(reports: List[Dict[str, Any]]) -> str:
    """Builds a small SVG line chart of overall_score over time by hand —
    no charting library dependency. Returns an empty-state message if fewer
    than 2 points exist, since a single point can't show a trend."""
    points = [(r.get("generated_at", ""), r.get("overall_score")) for r in reports if r.get("overall_score") is not None]
    if len(points) < 2:
        return ('<div class="empty-state">'
                '<span class="empty-icon">&#128200;</span>'
                '<p>Not enough history yet for a trend line.</p>'
                '<p class="muted">Run <code>quality_report.py</code> again after your next '
                'ingestion to start building one.</p></div>')

    width, height, pad_l, pad_r, pad_t, pad_b = 720, 220, 44, 24, 20, 32
    scores = [s for _, s in points]
    min_score, max_score = min(0, min(scores) - 5), max(100, max(scores) + 5)
    span = max(max_score - min_score, 1)
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    step_x = plot_w / max(len(points) - 1, 1)

    def y_for(score: float) -> float:
        return pad_t + plot_h - ((score - min_score) / span) * plot_h

    coords = [(pad_l + i * step_x, y_for(score)) for i, (_, score) in enumerate(points)]

    # Gridlines + y-axis labels at 0/25/50/75/100-ish ticks
    grid_lines = []
    tick_vals = [0, 25, 50, 75, 100]
    for tv in tick_vals:
        if tv < min_score or tv > max_score:
            continue
        y = y_for(tv)
        grid_lines.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" class="grid-line" />'
            f'<text x="{pad_l - 10}" y="{y + 4:.1f}" class="axis-label" text-anchor="end">{tv}</text>'
        )

    # Smooth-ish path using simple line segments (kept dependency-free)
    path_d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(coords))

    # Area fill under the line for visual polish
    area_d = path_d + f" L{coords[-1][0]:.1f},{pad_t + plot_h:.1f} L{coords[0][0]:.1f},{pad_t + plot_h:.1f} Z"

    circles = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" class="trend-point" fill="{_score_color(score)}">'
        f'<title>{_esc(_fmt_time(ts))}: {score:.1f}</title></circle>'
        for (ts, score), (x, y) in zip(points, coords)
    )

    x_labels = "".join(
        f'<text x="{x:.1f}" y="{height - 8}" class="axis-label" text-anchor="middle">'
        f'{_esc(_fmt_time(ts).split(chr(0x00b7))[0].strip())}</text>'
        for (ts, _), (x, _y) in zip(points, coords)
    )

    return f"""
    <svg viewBox="0 0 {width} {height}" class="trend-svg" role="img" aria-label="Corpus quality score over time">
      <defs>
        <linearGradient id="areaFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="var(--accent)" stop-opacity="0.28" />
          <stop offset="100%" stop-color="var(--accent)" stop-opacity="0" />
        </linearGradient>
      </defs>
      {''.join(grid_lines)}
      <path d="{area_d}" fill="url(#areaFill)" stroke="none" />
      <path d="{path_d}" fill="none" stroke="var(--accent)" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" />
      {circles}
      {x_labels}
    </svg>
    """


def _render_metrics_table(metrics: Dict[str, Dict[str, Any]]) -> str:
    rows = []
    for name, metric in metrics.items():
        score = metric.get("score")
        score_str = f"{score:.1f}" if score is not None else "N/A"
        cls = _score_class(score)
        label = name.replace("_", " ").title()
        pct = min(max(score or 0, 0), 100)
        rows.append(f"""
        <tr>
          <td class="metric-name">{_esc(label)}</td>
          <td class="metric-score">
            <div class="mini-bar-track"><div class="mini-bar-fill {cls}" style="width:{pct:.0f}%"></div></div>
            <span class="badge {cls}">{score_str}</span>
          </td>
          <td class="detail">{_esc(metric.get('detail', ''))}</td>
        </tr>""")
    return "\n".join(rows)


def _render_drift_banner(drift: Dict[str, Any]) -> str:
    status = drift.get("status")
    if status == "no_baseline":
        return '<div class="banner muted"><span class="banner-icon">&#8505;</span> No prior report to compare against yet — this was the first run.</div>'
    if status == "stable":
        compared = _esc(_fmt_time(drift.get("compared_against", "previous run")))
        return f'<div class="banner good"><span class="banner-icon">&#10003;</span> Stable — no metric regressed vs. {compared}.</div>'
    if status == "drift":
        compared = _esc(_fmt_time(drift.get("compared_against", "previous run")))
        items = "".join(
            f"<li>{_esc(r['metric'].replace('_', ' ').title())}: "
            f"{r['previous_score']} &rarr; {r['current_score']} "
            f"<span class=\"delta-bad\">({r['delta']:+.1f} pts)</span></li>"
            for r in drift.get("regressions", [])
        )
        return f'<div class="banner bad"><span class="banner-icon">&#9888;</span> Drift detected vs. {compared}:<ul>{items}</ul></div>'
    return ""


def _render_classification_section(report: Optional[Dict[str, Any]]) -> str:
    if report is None:
        return ('<div class="empty-state"><span class="empty-icon">&#129513;</span>'
                '<p>Not yet run.</p>'
                '<p class="muted">Generate this with:<br><code>python3 src/classification/eval_classifiers.py</code></p></div>')
    overall = report.get("overall_accuracy_pct")
    overall_str = f"{overall:.1f}%" if overall is not None else "N/A"
    cls = _score_class(overall)
    field_rows = ""
    for field, stats in report.get("field_accuracy", {}).items():
        total = stats.get("total", 0)
        correct = stats.get("correct", 0)
        pct_val = (100 * correct / total) if total else None
        pct = f"{pct_val:.1f}%" if pct_val is not None else "N/A"
        fcls = _score_class(pct_val)
        field_rows += (f'<tr><td class="metric-name">{_esc(field.replace("_", " ").title())}</td>'
                        f'<td><span class="badge {fcls} small">{pct}</span></td>'
                        f'<td class="detail">{correct}/{total}</td></tr>')
    return f"""
    <div class="stat-row">
      <span class="badge {cls} large">{overall_str}</span>
      <span class="stat-caption">exact-match accuracy &middot; {report.get('correct', 0)}/{report.get('total', 0)} documents fully correct</span>
    </div>
    <table><thead><tr><th>Field</th><th>Accuracy</th><th>Correct / Total</th></tr></thead>
    <tbody>{field_rows}</tbody></table>
    <p class="muted timestamp">Generated {_esc(_fmt_time(report.get('generated_at', 'unknown')))}</p>
    """


def _render_gold_section(report: Optional[Dict[str, Any]]) -> str:
    if report is None:
        return ('<div class="empty-state"><span class="empty-icon">&#129513;</span>'
                '<p>Not yet run.</p>'
                '<p class="muted">Generate this with:<br><code>python3 src/classification/score_against_gold.py '
                '--classified &lt;dir&gt; --labels &lt;gold_labels.jsonl&gt;</code></p></div>')
    overall = report.get("overall_accuracy_pct")
    overall_str = f"{overall:.1f}%" if overall is not None else "N/A"
    cls = _score_class(overall)
    field_rows = ""
    for field, stats in report.get("per_field_accuracy", {}).items():
        acc = stats.get("accuracy_pct")
        acc_str = f"{acc:.1f}%" if acc is not None else "N/A"
        fcls = _score_class(acc)
        field_rows += (f'<tr><td class="metric-name">{_esc(field.replace("_", " ").title())}</td>'
                        f'<td><span class="badge {fcls} small">{acc_str}</span></td>'
                        f'<td class="detail">{stats.get("correct", 0)}/{stats.get("total", 0)}</td></tr>')
    return f"""
    <div class="stat-row">
      <span class="badge {cls} large">{overall_str}</span>
      <span class="stat-caption">field accuracy vs. gold labels &middot; {report.get('total_scored', 0)} documents scored, {report.get('skipped_unclassified', 0)} skipped</span>
    </div>
    <table><thead><tr><th>Field</th><th>Accuracy</th><th>Correct / Total</th></tr></thead>
    <tbody>{field_rows}</tbody></table>
    <p class="muted timestamp">Generated {_esc(_fmt_time(report.get('generated_at', 'unknown')))}</p>
    """


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG Pipeline — Performance Dashboard</title>
<style>
  :root {{
    --bg: #0b0d12; --bg-glow: radial-gradient(1200px 600px at 10% -10%, rgba(91,141,239,0.12), transparent),
                              radial-gradient(900px 500px at 100% 0%, rgba(122,92,239,0.10), transparent);
    --card: #14161d; --card-border: #262a35; --card-hover: #171a23;
    --text: #eef0f4; --muted: #8b909c; --muted-2: #5c6270;
    --accent: #5b8def; --accent-2: #7a5cef;
    --good: #34c07a; --good-bg: rgba(52,192,122,0.14);
    --warn: #e0a736; --warn-bg: rgba(224,167,54,0.14);
    --bad: #ef5b52; --bad-bg: rgba(239,91,82,0.14);
    --border: #262a35;
    --radius: 14px;
    --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 8px 24px -8px rgba(0,0,0,0.5);
  }}
  * {{ box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    margin:0; padding:0 0 60px; background-color:var(--bg); background-image:var(--bg-glow);
    color:var(--text); font-family: "Inter", -apple-system, "Segoe UI", Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .top-bar {{
    padding: 36px 40px 28px; border-bottom: 1px solid var(--border);
    display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:16px;
  }}
  .brand {{ display:flex; align-items:center; gap:12px; }}
  .brand-mark {{
    width:38px; height:38px; border-radius:10px;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    display:flex; align-items:center; justify-content:center; font-size:18px; box-shadow: var(--shadow);
  }}
  h1 {{ font-size:20px; margin:0; font-weight:650; letter-spacing:-0.01em; }}
  .subtitle {{ color:var(--muted); margin:2px 0 0; font-size:13px; }}
  .refresh-hint {{ color:var(--muted-2); font-size:12px; text-align:right; }}
  .container {{ padding: 28px 40px 0; max-width:1280px; margin:0 auto; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  .card {{
    background:var(--card); border:1px solid var(--card-border); border-radius:var(--radius);
    padding:22px 24px; margin-bottom:20px; box-shadow: var(--shadow);
    transition: border-color 0.15s ease, transform 0.15s ease;
  }}
  .card:hover {{ border-color: #333a4a; }}
  .card h2 {{
    font-size:13px; margin:0 0 16px; color:var(--muted); font-weight:600;
    text-transform:uppercase; letter-spacing:0.06em; display:flex; align-items:center; gap:8px;
  }}
  .card h2 .dot {{ width:6px; height:6px; border-radius:50%; background:var(--accent); display:inline-block; }}

  .score-hero {{ font-size:56px; font-weight:750; letter-spacing:-0.02em; line-height:1; }}
  .score-hero.good {{ color:var(--good); }}
  .score-hero.warn {{ color:var(--warn); }}
  .score-hero.bad {{ color:var(--bad); }}
  .score-hero.muted {{ color:var(--muted); }}
  .score-sub {{ color:var(--muted); font-size:13px; margin-top:10px; }}

  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th, td {{ text-align:left; padding:10px 8px; border-bottom:1px solid var(--border); }}
  th {{ color:var(--muted-2); font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:0.05em; }}
  tr:last-child td {{ border-bottom:none; }}
  td.detail {{ color:var(--muted); }}
  td.metric-name {{ font-weight:500; }}
  td.metric-score {{ display:flex; align-items:center; gap:10px; width:220px; }}

  .mini-bar-track {{ flex:1; height:6px; border-radius:4px; background:#1e2129; overflow:hidden; }}
  .mini-bar-fill {{ height:100%; border-radius:4px; transition: width 0.4s ease; }}
  .mini-bar-fill.good {{ background: linear-gradient(90deg, var(--good), #5be0a0); }}
  .mini-bar-fill.warn {{ background: linear-gradient(90deg, var(--warn), #f0c675); }}
  .mini-bar-fill.bad {{ background: linear-gradient(90deg, var(--bad), #f28b84); }}
  .mini-bar-fill.muted {{ background: var(--muted-2); }}

  .badge {{
    display:inline-flex; align-items:center; justify-content:center; padding:3px 10px;
    border-radius:999px; font-weight:700; font-size:12px; min-width:52px;
  }}
  .badge.good {{ background:var(--good-bg); color:var(--good); }}
  .badge.warn {{ background:var(--warn-bg); color:var(--warn); }}
  .badge.bad {{ background:var(--bad-bg); color:var(--bad); }}
  .badge.muted {{ background:rgba(139,144,156,0.12); color:var(--muted); }}
  .badge.large {{ font-size:22px; padding:6px 18px; min-width:0; }}
  .badge.small {{ font-size:11px; padding:2px 8px; min-width:0; }}

  .stat-row {{ display:flex; align-items:center; gap:14px; margin-bottom:16px; flex-wrap:wrap; }}
  .stat-caption {{ color:var(--muted); font-size:13px; }}

  .banner {{ padding:14px 16px; border-radius:10px; margin-bottom:6px; font-size:13px; display:flex; gap:10px; align-items:flex-start; }}
  .banner-icon {{ flex-shrink:0; }}
  .banner.good {{ background:var(--good-bg); border:1px solid rgba(52,192,122,0.35); color:#d3f5e3; }}
  .banner.bad {{ background:var(--bad-bg); border:1px solid rgba(239,91,82,0.35); color:#fbd8d5; }}
  .banner.muted {{ background:rgba(139,144,156,0.08); border:1px solid var(--border); color:var(--muted); }}
  .banner ul {{ margin:8px 0 0 18px; padding:0; }}
  .delta-bad {{ color:var(--bad); font-weight:600; }}

  .muted {{ color:var(--muted); font-size:13px; }}
  .timestamp {{ margin-top:14px; font-size:12px; color:var(--muted-2); }}
  code {{ background:#1c1f28; padding:2px 6px; border-radius:5px; font-size:12px; color:#c9d1ff; }}

  .empty-state {{ text-align:center; padding:28px 10px; }}
  .empty-icon {{ font-size:28px; display:block; margin-bottom:10px; opacity:0.7; }}
  .empty-state p {{ margin:4px 0; }}

  .trend-svg {{ width:100%; height:auto; }}
  .grid-line {{ stroke:var(--border); stroke-width:1; }}
  .axis-label {{ fill:var(--muted-2); font-size:10px; font-family: inherit; }}
  .trend-point {{ stroke:var(--card); stroke-width:2; }}

  .full-width {{ grid-column: 1 / -1; }}

  footer {{ text-align:center; color:var(--muted-2); font-size:12px; padding:32px 0 0; }}

  @media (max-width: 860px) {{
    .grid {{ grid-template-columns:1fr; }}
    .top-bar, .container {{ padding-left:20px; padding-right:20px; }}
  }}
</style>
</head>
<body>
  <div class="top-bar">
    <div class="brand">
      <div class="brand-mark">&#128202;</div>
      <div>
        <h1>RAG Pipeline &mdash; Performance Dashboard</h1>
        <p class="subtitle">Enterprise Knowledge Classification &amp; Retrieval Pipeline</p>
      </div>
    </div>
    <div class="refresh-hint">
      Generated {generated_at}<br>
      regenerate with <code>python3 src/qa/dashboard.py</code>
    </div>
  </div>

  <div class="container">
    <div class="grid">
      <div class="card">
        <h2><span class="dot"></span>Corpus Quality Score</h2>
        <div class="score-hero {hero_class}">{hero_score}</div>
        <p class="score-sub">as of {latest_report_time}</p>
      </div>
      <div class="card">
        <h2><span class="dot"></span>Drift Status</h2>
        {drift_banner}
      </div>

      <div class="card full-width">
        <h2><span class="dot"></span>Score Trend</h2>
        {trend_svg}
      </div>

      <div class="card full-width">
        <h2><span class="dot"></span>Latest Corpus Quality Breakdown</h2>
        <table>
          <thead><tr><th>Metric</th><th>Score</th><th>Detail</th></tr></thead>
          <tbody>{metrics_rows}</tbody>
        </table>
      </div>

      <div class="card">
        <h2><span class="dot"></span>Classification Accuracy &mdash; Labeled Test Set</h2>
        {classification_section}
      </div>

      <div class="card">
        <h2><span class="dot"></span>Classification Accuracy &mdash; Gold-Labeled Corpus</h2>
        {gold_section}
      </div>
    </div>

    <footer>Timsoft Internship Project 3 &middot; From Atlas to RAG</footer>
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
        hero_class = _score_class(hero_score)
        drift_banner = _render_drift_banner(latest.get("drift", {"status": "no_baseline"}))
        metrics_rows = _render_metrics_table(latest.get("metrics", {}))
        latest_time = _esc(_fmt_time(latest.get("generated_at", "unknown")))
    else:
        hero_str, hero_class = "N/A", "muted"
        drift_banner = ('<div class="banner muted"><span class="banner-icon">&#8505;</span> No quality report found yet. Generate one with: '
                         '<code>python3 -m src.qa.quality_report</code></div>')
        metrics_rows = '<tr><td colspan="3" class="muted">No data yet.</td></tr>'
        latest_time = "never"

    return _TEMPLATE.format(
        generated_at=_esc(_fmt_time(datetime.now(timezone.utc).isoformat())),
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
