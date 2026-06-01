#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime
from typing import List, Dict, Any
from utils import (
    setup_logger, load_goal, load_strategy, REPORTS_DIR,
    read_jsonl, REFLECTIONS_FILE, now_iso
)
from trade_logger import load_all_trades, load_recent_trades
from scoring_engine import score, score_by_regime
from version_manager import load_all_versions
from reflection_engine import run_reflection

logger = setup_logger("report_generator")


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def _fmt_float(v: float, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}"


def generate_report() -> str:
    goal = load_goal()
    strategy = load_strategy()
    all_trades = load_all_trades()
    recent_trades = load_recent_trades(20)
    reflections = read_jsonl(REFLECTIONS_FILE)
    versions = load_all_versions()

    perf = score(all_trades, goal) if all_trades else {
        "score": 0, "sharpe": 0, "max_drawdown": 0, "return": 0, "win_rate": 0, "trade_count": 0
    }
    regime_perf = score_by_regime(all_trades, goal) if all_trades else {}
    hypothesis = run_reflection(recent_trades, strategy, goal) if len(recent_trades) >= 3 else None

    lines = []
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append(f"# Local Self-Improving Trading Engine — Research Report")
    lines.append(f"\n**Generated:** {ts}\n")
    lines.append("---\n")

    lines.append("## Current Strategy\n")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| Version | v{strategy.get('version', '01')} |")
    lines.append(f"| Asset | {goal.get('asset', 'BTC/USDT')} |")
    lines.append(f"| Direction | {strategy.get('direction', 'LONG')} |")
    lines.append(f"| Indicator | {strategy['entry']['indicator']} |")
    lines.append(f"| Threshold | {strategy['entry']['threshold']} |")
    lines.append(f"| Stop Loss | {strategy.get('stop_loss_pct', 2)}% |")
    lines.append(f"| Take Profit | {strategy.get('take_profit_pct', 4)}% |")
    lines.append(f"| Position Size | {strategy.get('position_size_pct', 5)}% |")
    lines.append(f"| Timeframe | {strategy.get('timeframe', '1h')} |")
    lines.append("")

    lines.append("## Current Performance\n")
    lines.append(f"| Metric | Value | Target |")
    lines.append(f"|--------|-------|--------|")
    lines.append(f"| Composite Score | {_fmt_float(perf['score'])} | > 0.0 |")
    lines.append(f"| Total Return | {_fmt_pct(perf['return'])} | >= {_fmt_pct(goal.get('target_return_30d', 0.05))} |")
    lines.append(f"| Max Drawdown | {_fmt_pct(perf['max_drawdown'])} | <= {_fmt_pct(goal.get('max_drawdown', 0.08))} |")
    lines.append(f"| Sharpe Ratio | {_fmt_float(perf['sharpe'])} | >= {goal.get('min_sharpe', 1.2)} |")
    lines.append(f"| Win Rate | {_fmt_pct(perf['win_rate'])} | — |")
    lines.append(f"| Total Trades | {perf.get('trade_count', 0)} | — |")
    lines.append("")

    lines.append("## Recent Trades (Last 20)\n")
    if recent_trades:
        lines.append("| # | Timestamp | Entry | Exit | Return | PnL | Drawdown | Regime | Exit Reason |")
        lines.append("|---|-----------|-------|------|--------|-----|----------|--------|-------------|")
        for i, t in enumerate(recent_trades[-20:], 1):
            ts_str = t.get("timestamp", "")[:19]
            lines.append(
                f"| {i} | {ts_str} | {t.get('entry_price', 0):.2f} | {t.get('exit_price', 0):.2f} "
                f"| {_fmt_pct(t.get('return', 0))} | {t.get('pnl', 0):.2f} "
                f"| {_fmt_pct(t.get('drawdown', 0))} | {t.get('regime', '-')} | {t.get('exit_reason', '-')} |"
            )
    else:
        lines.append("_No trades recorded yet._")
    lines.append("")

    lines.append("## Market Regimes\n")
    if regime_perf:
        lines.append("| Regime | Score | Return | Sharpe | Max DD | Win Rate | Trades |")
        lines.append("|--------|-------|--------|--------|--------|----------|--------|")
        for regime, rp in sorted(regime_perf.items()):
            lines.append(
                f"| {regime} | {rp['score']:.3f} | {_fmt_pct(rp['return'])} "
                f"| {rp['sharpe']:.3f} | {_fmt_pct(rp['max_drawdown'])} "
                f"| {_fmt_pct(rp['win_rate'])} | {rp.get('trade_count', 0)} |"
            )
    else:
        lines.append("_No regime data available yet._")
    lines.append("")

    lines.append("## Strengths\n")
    strengths = []
    if perf["return"] >= goal.get("target_return_30d", 0.05):
        strengths.append(f"- Return of {_fmt_pct(perf['return'])} meets the target of {_fmt_pct(goal.get('target_return_30d', 0.05))}.")
    if perf["max_drawdown"] <= goal.get("max_drawdown", 0.08):
        strengths.append(f"- Drawdown of {_fmt_pct(perf['max_drawdown'])} is within the {_fmt_pct(goal.get('max_drawdown', 0.08))} limit.")
    if perf["sharpe"] >= goal.get("min_sharpe", 1.2):
        strengths.append(f"- Sharpe ratio of {perf['sharpe']:.3f} exceeds minimum of {goal.get('min_sharpe', 1.2)}.")
    if perf["win_rate"] > 0.5:
        strengths.append(f"- Win rate of {_fmt_pct(perf['win_rate'])} is above 50%.")
    best_regime = max(regime_perf.items(), key=lambda x: x[1]["score"], default=(None, None))
    if best_regime[0]:
        strengths.append(f"- Best performance in {best_regime[0]} regime (score: {best_regime[1]['score']:.3f}).")
    if not strengths:
        strengths.append("- Insufficient data to identify strengths.")
    lines.extend(strengths)
    lines.append("")

    lines.append("## Weaknesses\n")
    weaknesses = []
    if perf["return"] < goal.get("target_return_30d", 0.05):
        weaknesses.append(f"- Return {_fmt_pct(perf['return'])} is below target {_fmt_pct(goal.get('target_return_30d', 0.05))}.")
    if perf["max_drawdown"] > goal.get("max_drawdown", 0.08):
        weaknesses.append(f"- Drawdown {_fmt_pct(perf['max_drawdown'])} exceeds limit {_fmt_pct(goal.get('max_drawdown', 0.08))}.")
    if perf["sharpe"] < goal.get("min_sharpe", 1.2):
        weaknesses.append(f"- Sharpe {perf['sharpe']:.3f} is below minimum {goal.get('min_sharpe', 1.2)}.")
    if perf["win_rate"] <= 0.5 and perf.get("trade_count", 0) > 0:
        weaknesses.append(f"- Win rate of {_fmt_pct(perf['win_rate'])} is at or below 50%.")
    worst_regime = min(regime_perf.items(), key=lambda x: x[1]["score"], default=(None, None))
    if worst_regime[0] and worst_regime[1]["score"] < 0:
        weaknesses.append(f"- Poor performance in {worst_regime[0]} regime (score: {worst_regime[1]['score']:.3f}).")
    if not weaknesses:
        weaknesses.append("- No major weaknesses detected.")
    lines.extend(weaknesses)
    lines.append("")

    lines.append("## Hypotheses\n")
    if reflections:
        for i, r in enumerate(reflections[-5:], 1):
            lines.append(f"### Hypothesis {i}")
            lines.append(f"- **Timestamp:** {r.get('timestamp', '')}")
            lines.append(f"- **Variable Changed:** `{r.get('changed_variable', '')}`")
            lines.append(f"- **Change:** {r.get('old_value', '')} → {r.get('new_value', '')}")
            lines.append(f"- **Reason:** {r.get('reason', '')}")
            lines.append(f"- **Expected Effect:** {r.get('expected_effect', '')}")
            lines.append("")
    else:
        lines.append("_No hypotheses generated yet._\n")

    lines.append("## Recommended Next Change\n")
    if hypothesis:
        lines.append(f"| Field | Value |")
        lines.append(f"|-------|-------|")
        lines.append(f"| Variable | `{hypothesis['changed_variable']}` |")
        lines.append(f"| Current Value | {hypothesis['old_value']} |")
        lines.append(f"| Proposed Value | {hypothesis['new_value']} |")
        lines.append(f"| Primary Weakness | {hypothesis['weakness']} |")
        lines.append(f"| Reason | {hypothesis['reason']} |")
        lines.append(f"| Expected Effect | {hypothesis['expected_effect']} |")
    else:
        lines.append("_Insufficient trade data to generate recommendation (need >= 3 trades)._")
    lines.append("")

    lines.append("## Expected Improvement\n")
    if hypothesis:
        var = hypothesis["changed_variable"]
        improvement_estimates = {
            "rsi_threshold": "Potentially improved entry timing; estimated 5-15% improvement in win rate.",
            "stop_loss_pct": "Reduced per-trade drawdown; expected 10-20% reduction in max drawdown.",
            "take_profit_pct": "Larger average winners; expected 5-15% improvement in total return.",
            "position_size_pct": "Adjusted risk exposure; impact on drawdown and return proportional to size change.",
        }
        lines.append(improvement_estimates.get(var, "Improvement estimate not available."))
    else:
        lines.append("_No improvement estimate available._")
    lines.append("")

    lines.append("## Confidence\n")
    n_trades = perf.get("trade_count", 0)
    if n_trades < 5:
        confidence = "LOW"
        conf_reason = f"Only {n_trades} trades. Need 20+ for statistical significance."
    elif n_trades < 20:
        confidence = "MEDIUM"
        conf_reason = f"{n_trades} trades available. Patterns emerging but more data needed."
    else:
        confidence = "HIGH"
        conf_reason = f"{n_trades} trades provide reasonable statistical basis."
    lines.append(f"**{confidence}** — {conf_reason}")
    lines.append("")

    lines.append("## Strategy Version History\n")
    if versions:
        lines.append("| Version | Indicator | Threshold | SL% | TP% | Size% |")
        lines.append("|---------|-----------|-----------|-----|-----|-------|")
        for v in versions:
            lines.append(
                f"| v{v.get('version', '?')} | {v.get('entry', {}).get('indicator', '?')} "
                f"| {v.get('entry', {}).get('threshold', '?')} "
                f"| {v.get('stop_loss_pct', '?')} | {v.get('take_profit_pct', '?')} "
                f"| {v.get('position_size_pct', '?')} |"
            )
    else:
        lines.append("_No version history yet._")
    lines.append("")

    lines.append("---\n")
    lines.append(f"_Report generated by Local Self-Improving Trading Engine at {ts}_")

    return "\n".join(lines)


def save_report(content: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"report_{ts}.md"
    with open(path, "w") as f:
        f.write(content)
    logger.info(f"Report saved: {path}")
    return path


if __name__ == "__main__":
    report = generate_report()
    path = save_report(report)
    print(report)
    print(f"\nReport saved to: {path}")
