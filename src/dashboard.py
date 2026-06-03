#!/usr/bin/env python3
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

GITHUB_REPO = os.getenv("GITHUB_REPO") or st.secrets.get("GITHUB_REPO", "")

if GITHUB_REPO:
    from github_reader import read_jsonl, read_yaml, list_version_yamls
    def _load_goal():       return read_yaml(GITHUB_REPO, "configs/goal.yaml")
    def _load_strategy():   return read_yaml(GITHUB_REPO, "configs/strategy.yaml")
    def _load_trades():     return read_jsonl(GITHUB_REPO, "data/trades.jsonl")
    def _load_reflections():return read_jsonl(GITHUB_REPO, "data/reflections.jsonl")
    def _load_versions():   return list_version_yamls(GITHUB_REPO)
else:
    from utils import load_goal, load_strategy
    from trade_logger import load_all_trades
    from reflection_engine import load_reflections
    from version_manager import load_all_versions
    def _load_goal():       return load_goal()
    def _load_strategy():   return load_strategy()
    def _load_trades():     return load_all_trades()
    def _load_reflections():return load_reflections()
    def _load_versions():   return load_all_versions()

from scoring_engine import score, score_by_regime

st.set_page_config(
    page_title="Trading Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

REFRESH_INTERVAL = 60


@st.cache_data(ttl=REFRESH_INTERVAL)
def get_data():
    goal        = _load_goal()
    strategy    = _load_strategy()
    all_trades  = _load_trades()
    reflections = _load_reflections()
    versions    = _load_versions()

    if not goal:
        goal = {"asset": "BTC/USDT", "target_return_30d": 0.05,
                "max_drawdown": 0.08, "min_sharpe": 1.2}

    perf = score(all_trades, goal) if all_trades else {
        "score": 0, "sharpe": 0, "max_drawdown": 0, "return": 0,
        "win_rate": 0, "trade_count": 0, "return_score": 0,
        "dd_score": 0, "sharpe_score": 0,
    }
    regime_perf = score_by_regime(all_trades, goal) if all_trades else {}
    return goal, strategy, all_trades, reflections, versions, perf, regime_perf


# ── charts ─────────────────────────────────────────────────────────────────

def _equity_fig(trades):
    if not trades:
        return go.Figure()
    df = pd.DataFrame(trades)
    df["cum_return"] = (1 + df["return"]).cumprod() - 1
    df["equity"] = 10000 * (1 + df["cum_return"])
    df["trade_num"] = range(1, len(df) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["trade_num"], y=df["equity"], mode="lines",
        name="Equity", line=dict(color="#00ff88", width=2),
        hovertemplate="Trade #%{x}<br>Equity: $%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(title="Equity Curve", xaxis_title="Trade #",
                      yaxis_title="Portfolio ($)", template="plotly_dark", height=350)
    return fig


def _drawdown_fig(trades):
    if not trades:
        return go.Figure()
    df = pd.DataFrame(trades)
    cum = (1 + df["return"]).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax() * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(df) + 1)), y=dd, mode="lines",
        fill="tozeroy", name="Drawdown",
        line=dict(color="#ff4444", width=2),
        hovertemplate="Trade #%{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(title="Drawdown", xaxis_title="Trade #",
                      yaxis_title="Drawdown (%)", template="plotly_dark", height=280)
    return fig


def _return_dist_fig(trades):
    if not trades:
        return go.Figure()
    returns = [t["return"] * 100 for t in trades]
    fig = px.histogram(x=returns, nbins=30, template="plotly_dark",
                       color_discrete_sequence=["#4488ff"],
                       labels={"x": "Return (%)"})
    fig.update_layout(title="Return Distribution", height=280)
    return fig


def _regime_fig(regime_perf):
    if not regime_perf:
        return go.Figure()
    regimes = list(regime_perf.keys())
    scores  = [regime_perf[r]["score"] for r in regimes]
    colors  = ["#00ff88" if s > 0 else "#ff4444" for s in scores]
    fig = go.Figure(go.Bar(x=regimes, y=scores, marker_color=colors,
                           text=[f"{s:.3f}" for s in scores], textposition="outside"))
    fig.update_layout(title="Score by Regime", template="plotly_dark", height=280)
    return fig


def _cumulative_returns_fig(trades):
    if not trades:
        return go.Figure()
    df = pd.DataFrame(trades)
    df["trade_num"] = range(1, len(df) + 1)
    df["cum_return_pct"] = ((1 + df["return"]).cumprod() - 1) * 100
    df["return_pct"] = df["return"] * 100
    colors = ["#00ff88" if r >= 0 else "#ff4444" for r in df["return_pct"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["trade_num"], y=df["return_pct"], name="Per-Trade Return",
        marker_color=colors, opacity=0.6,
        hovertemplate="Trade #%{x}<br>Return: %{y:.3f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["trade_num"], y=df["cum_return_pct"], name="Cumulative Return",
        mode="lines", line=dict(color="#ffdd00", width=2.5),
        hovertemplate="Trade #%{x}<br>Cumulative: %{y:.3f}%<extra></extra>",
        yaxis="y2",
    ))
    fig.update_layout(
        title="Trade Returns & Cumulative Performance",
        xaxis_title="Trade #",
        yaxis=dict(title="Per-Trade Return (%)", showgrid=False),
        yaxis2=dict(title="Cumulative Return (%)", overlaying="y", side="right"),
        template="plotly_dark",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _trade_timeline_fig(trades):
    if not trades:
        return go.Figure()
    df = pd.DataFrame(trades)
    df["color"] = df["return"].apply(lambda r: "#00ff88" if r >= 0 else "#ff4444")
    df["result"] = df["return"].apply(lambda r: "Win" if r >= 0 else "Loss")
    df["return_pct"] = (df["return"] * 100).round(3)
    df["trade_num"] = range(1, len(df) + 1)

    wins   = df[df["return"] >= 0]
    losses = df[df["return"] < 0]

    fig = go.Figure()
    if not wins.empty:
        fig.add_trace(go.Scatter(
            x=wins["trade_num"], y=wins["return_pct"], mode="markers",
            name="Win", marker=dict(color="#00ff88", size=8, symbol="circle"),
            hovertemplate="Trade #%{x}<br>Return: %{y:.3f}%<extra>Win</extra>",
        ))
    if not losses.empty:
        fig.add_trace(go.Scatter(
            x=losses["trade_num"], y=losses["return_pct"], mode="markers",
            name="Loss", marker=dict(color="#ff4444", size=8, symbol="circle"),
            hovertemplate="Trade #%{x}<br>Return: %{y:.3f}%<extra>Loss</extra>",
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title="Per-Trade Returns (Win / Loss)",
        xaxis_title="Trade #", yaxis_title="Return (%)",
        template="plotly_dark", height=320,
    )
    return fig


# ── main ───────────────────────────────────────────────────────────────────

def main():
    goal, strategy, all_trades, reflections, versions, perf, regime_perf = get_data()

    st.sidebar.title("Trading Engine")
    st.sidebar.markdown(f"**Asset:** {goal.get('asset', 'BTC/USDT')}")
    st.sidebar.markdown(f"**Strategy:** v{strategy.get('version', '01')}")
    st.sidebar.markdown(f"**Total Trades:** {len(all_trades)}")
    st.sidebar.markdown(f"**Refreshed:** {datetime.utcnow().strftime('%H:%M:%S UTC')}")
    if st.sidebar.button("Refresh Now"):
        st.cache_data.clear()
        st.rerun()

    page = st.sidebar.radio("Navigate", [
        "Overview", "Trade History & Returns",
        "Performance", "Reflection History", "Strategy Versions",
    ])

    # ── Overview ──────────────────────────────────────────────────────────
    if page == "Overview":
        st.title("Trading Engine Overview")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Score",    f"{perf['score']:.3f}")
        c2.metric("Return",   f"{perf['return']*100:.2f}%",
                  delta=f"Target {goal.get('target_return_30d',0.05)*100:.1f}%")
        c3.metric("Sharpe",   f"{perf['sharpe']:.3f}",
                  delta=f"Min {goal.get('min_sharpe',1.2)}")
        c4.metric("Max DD",   f"{perf['max_drawdown']*100:.2f}%")
        c5.metric("Win Rate", f"{perf['win_rate']*100:.1f}%")

        st.markdown("---")
        st.plotly_chart(_equity_fig(all_trades), use_container_width=True)

        cl, cr = st.columns(2)
        with cl:
            st.plotly_chart(_drawdown_fig(all_trades), use_container_width=True)
        with cr:
            st.plotly_chart(_regime_fig(regime_perf), use_container_width=True)

        st.subheader("Current Strategy")
        sc1, sc2 = st.columns(2)
        with sc1:
            st.json({"version": strategy.get("version"),
                     "indicator": strategy.get("entry", {}).get("indicator"),
                     "threshold": strategy.get("entry", {}).get("threshold"),
                     "direction": strategy.get("direction")})
        with sc2:
            st.json({"stop_loss_pct": strategy.get("stop_loss_pct"),
                     "take_profit_pct": strategy.get("take_profit_pct"),
                     "position_size_pct": strategy.get("position_size_pct"),
                     "timeframe": strategy.get("timeframe")})

    # ── Trade History & Returns ────────────────────────────────────────────
    elif page == "Trade History & Returns":
        st.title("Trade History & Returns")

        if not all_trades:
            st.info("No trades yet. The engine logs trades every hour.")
        else:
            st.plotly_chart(_cumulative_returns_fig(all_trades), use_container_width=True)
            st.plotly_chart(_trade_timeline_fig(all_trades), use_container_width=True)

            st.markdown("---")
            st.subheader(f"All Trades ({len(all_trades)} total)")

            df = pd.DataFrame(all_trades)
            df["return_%"] = (df["return"] * 100).round(3)
            df["drawdown_%"] = (df["drawdown"] * 100).round(3)
            show_cols = ["timestamp","asset","entry_price","exit_price",
                         "return_%","pnl","drawdown_%","hold_time",
                         "regime","strategy_version","exit_reason"]
            existing = [c for c in show_cols if c in df.columns]

            def _highlight(row):
                color = "background-color: #1a3a1a" if row.get("return_%", 0) >= 0 \
                        else "background-color: #3a1a1a"
                return [color] * len(row)

            st.dataframe(
                df[existing].sort_values("timestamp", ascending=False)
                  .style.apply(_highlight, axis=1),
                use_container_width=True,
            )

            st.markdown("---")
            st.subheader("Summary Stats")
            s1, s2, s3, s4 = st.columns(4)
            wins   = df[df["return"] >= 0]
            losses = df[df["return"] < 0]
            s1.metric("Total Trades", len(df))
            s2.metric("Wins / Losses", f"{len(wins)} / {len(losses)}")
            s3.metric("Avg Win",  f"{wins['return'].mean()*100:.3f}%" if len(wins)  else "—")
            s4.metric("Avg Loss", f"{losses['return'].mean()*100:.3f}%" if len(losses) else "—")

            if "hold_time" in df.columns:
                s5, s6, s7, s8 = st.columns(4)
                s5.metric("Avg Hold (bars)", f"{df['hold_time'].mean():.1f}")
                s6.metric("Best Trade",  f"{df['return'].max()*100:.3f}%")
                s7.metric("Worst Trade", f"{df['return'].min()*100:.3f}%")
                s8.metric("Total PnL",   f"${df['pnl'].sum():.2f}" if "pnl" in df else "—")

            if "regime" in df.columns:
                st.markdown("---")
                st.subheader("Trades by Regime")
                regime_counts = df["regime"].value_counts().reset_index()
                regime_counts.columns = ["Regime", "Count"]
                fig = px.bar(regime_counts, x="Regime", y="Count",
                             template="plotly_dark", color="Count",
                             color_continuous_scale="viridis")
                fig.update_layout(height=280, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    # ── Performance ───────────────────────────────────────────────────────
    elif page == "Performance":
        st.title("Performance Analytics")

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Return Score",   f"{perf.get('return_score',0):.3f}", help="40% weight")
        sc2.metric("Drawdown Score", f"{perf.get('dd_score',0):.3f}",     help="30% weight")
        sc3.metric("Sharpe Score",   f"{perf.get('sharpe_score',0):.3f}", help="30% weight")

        st.plotly_chart(_equity_fig(all_trades), use_container_width=True)

        cl, cr = st.columns(2)
        with cl:
            st.plotly_chart(_drawdown_fig(all_trades), use_container_width=True)
        with cr:
            st.plotly_chart(_return_dist_fig(all_trades), use_container_width=True)

        st.subheader("Regime Performance")
        if regime_perf:
            rdf = pd.DataFrame([{
                "Regime": k, "Score": f"{v['score']:.3f}",
                "Return": f"{v['return']*100:.2f}%",
                "Sharpe": f"{v['sharpe']:.3f}",
                "Max DD": f"{v['max_drawdown']*100:.2f}%",
                "Win Rate": f"{v['win_rate']*100:.1f}%",
                "Trades": v.get("trade_count", 0),
            } for k, v in sorted(regime_perf.items())])
            st.dataframe(rdf, use_container_width=True)
        else:
            st.info("No regime data yet.")

    # ── Reflection History ────────────────────────────────────────────────
    elif page == "Reflection History":
        st.title("Reflection & Optimization History")
        if reflections:
            for i, r in enumerate(reversed(reflections), 1):
                label = (f"#{len(reflections)-i+1} — "
                         f"{r.get('timestamp','')[:19]} | "
                         f"`{r.get('changed_variable','?')}`: "
                         f"{r.get('old_value','?')} → {r.get('new_value','?')}")
                with st.expander(label):
                    st.markdown(f"**Variable:** `{r.get('changed_variable','')}`")
                    st.markdown(f"**Change:** {r.get('old_value','')} → {r.get('new_value','')}")
                    st.markdown(f"**Reason:** {r.get('reason','')}")
                    st.markdown(f"**Expected Effect:** {r.get('expected_effect','')}")
                    st.markdown(f"**New Version:** v{r.get('version','?')}")
        else:
            st.info("No reflections yet. Triggers after every 5 trades.")

    # ── Strategy Versions ─────────────────────────────────────────────────
    elif page == "Strategy Versions":
        st.title("Strategy Version History")
        if versions:
            for v in reversed(versions):
                with st.expander(f"v{v.get('version','?')} — {v.get('_file','')}"):
                    st.json({k: val for k, val in v.items() if not k.startswith("_")})
        else:
            st.info("No version snapshots yet.")
        st.markdown("---")
        st.subheader("Current Active Strategy")
        st.json(strategy)


if __name__ == "__main__":
    main()
