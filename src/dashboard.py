#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from utils import load_goal, load_strategy
from trade_logger import load_all_trades, load_recent_trades
from scoring_engine import score, score_by_regime
from version_manager import load_all_versions
from reflection_engine import load_reflections

st.set_page_config(
    page_title="Trading Engine Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

REFRESH_INTERVAL = 60


@st.cache_data(ttl=REFRESH_INTERVAL)
def get_data():
    goal = load_goal()
    strategy = load_strategy()
    all_trades = load_all_trades()
    reflections = load_reflections()
    versions = load_all_versions()
    perf = score(all_trades, goal) if all_trades else {
        "score": 0, "sharpe": 0, "max_drawdown": 0, "return": 0, "win_rate": 0, "trade_count": 0
    }
    regime_perf = score_by_regime(all_trades, goal) if all_trades else {}
    return goal, strategy, all_trades, reflections, versions, perf, regime_perf


def build_equity_curve(trades):
    if not trades:
        return go.Figure()
    df = pd.DataFrame(trades)
    df["cumulative_return"] = (1 + df["return"]).cumprod() - 1
    df["equity"] = 10000 * (1 + df["cumulative_return"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df["equity"],
        mode="lines",
        name="Equity",
        line=dict(color="#00ff88", width=2),
    ))
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Trade #",
        yaxis_title="Portfolio Value ($)",
        template="plotly_dark",
        height=350,
    )
    return fig


def build_drawdown_curve(trades):
    if not trades:
        return go.Figure()
    df = pd.DataFrame(trades)
    cum = (1 + df["return"]).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=dd * 100,
        mode="lines",
        fill="tozeroy",
        name="Drawdown",
        line=dict(color="#ff4444", width=2),
    ))
    fig.update_layout(
        title="Drawdown Curve",
        xaxis_title="Trade #",
        yaxis_title="Drawdown (%)",
        template="plotly_dark",
        height=300,
    )
    return fig


def build_return_distribution(trades):
    if not trades:
        return go.Figure()
    returns = [t["return"] * 100 for t in trades]
    fig = px.histogram(
        x=returns,
        nbins=30,
        title="Return Distribution (%)",
        labels={"x": "Return (%)"},
        template="plotly_dark",
        color_discrete_sequence=["#4488ff"],
    )
    fig.update_layout(height=300)
    return fig


def build_regime_chart(regime_perf):
    if not regime_perf:
        return go.Figure()
    regimes = list(regime_perf.keys())
    scores = [regime_perf[r]["score"] for r in regimes]
    colors = ["#00ff88" if s > 0 else "#ff4444" for s in scores]
    fig = go.Figure(go.Bar(
        x=regimes,
        y=scores,
        marker_color=colors,
        text=[f"{s:.3f}" for s in scores],
        textposition="outside",
    ))
    fig.update_layout(
        title="Performance Score by Regime",
        xaxis_title="Regime",
        yaxis_title="Score",
        template="plotly_dark",
        height=300,
    )
    return fig


def main():
    goal, strategy, all_trades, reflections, versions, perf, regime_perf = get_data()

    st.sidebar.title("Trading Engine")
    st.sidebar.markdown(f"**Asset:** {goal.get('asset', 'BTC/USDT')}")
    st.sidebar.markdown(f"**Strategy v{strategy.get('version', '01')}**")
    st.sidebar.markdown(f"**Trades:** {len(all_trades)}")
    st.sidebar.markdown(f"**Last refresh:** {datetime.utcnow().strftime('%H:%M:%S UTC')}")

    if st.sidebar.button("Refresh Now"):
        st.cache_data.clear()
        st.rerun()

    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Performance", "Trade History", "Reflection History", "Strategy Versions"],
    )

    if page == "Overview":
        st.title("Trading Engine Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Score", f"{perf['score']:.3f}")
        col2.metric("Return", f"{perf['return'] * 100:.2f}%", delta=f"Target: {goal.get('target_return_30d', 0.05)*100:.1f}%")
        col3.metric("Sharpe", f"{perf['sharpe']:.3f}", delta=f"Min: {goal.get('min_sharpe', 1.2)}")
        col4.metric("Max DD", f"{perf['max_drawdown'] * 100:.2f}%")
        col5.metric("Win Rate", f"{perf['win_rate'] * 100:.1f}%")

        st.markdown("---")
        col_l, col_r = st.columns(2)
        with col_l:
            st.plotly_chart(build_equity_curve(all_trades), use_container_width=True)
        with col_r:
            st.plotly_chart(build_drawdown_curve(all_trades), use_container_width=True)

        col_l2, col_r2 = st.columns(2)
        with col_l2:
            st.plotly_chart(build_return_distribution(all_trades), use_container_width=True)
        with col_r2:
            st.plotly_chart(build_regime_chart(regime_perf), use_container_width=True)

        st.subheader("Current Strategy")
        sc1, sc2 = st.columns(2)
        with sc1:
            st.json({
                "version": strategy.get("version"),
                "indicator": strategy["entry"]["indicator"],
                "threshold": strategy["entry"]["threshold"],
                "direction": strategy.get("direction"),
            })
        with sc2:
            st.json({
                "stop_loss_pct": strategy.get("stop_loss_pct"),
                "take_profit_pct": strategy.get("take_profit_pct"),
                "position_size_pct": strategy.get("position_size_pct"),
                "timeframe": strategy.get("timeframe"),
            })

    elif page == "Performance":
        st.title("Performance Analytics")
        st.subheader("Score Breakdown")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Return Score", f"{perf.get('return_score', 0):.3f}", help="40% weight")
        sc2.metric("Drawdown Score", f"{perf.get('dd_score', 0):.3f}", help="30% weight")
        sc3.metric("Sharpe Score", f"{perf.get('sharpe_score', 0):.3f}", help="30% weight")

        st.markdown("---")
        st.subheader("Equity Curve")
        st.plotly_chart(build_equity_curve(all_trades), use_container_width=True)
        st.subheader("Drawdown")
        st.plotly_chart(build_drawdown_curve(all_trades), use_container_width=True)

        st.subheader("Regime Performance")
        if regime_perf:
            regime_df = pd.DataFrame([
                {
                    "Regime": k,
                    "Score": f"{v['score']:.3f}",
                    "Return": f"{v['return']*100:.2f}%",
                    "Sharpe": f"{v['sharpe']:.3f}",
                    "Max DD": f"{v['max_drawdown']*100:.2f}%",
                    "Win Rate": f"{v['win_rate']*100:.1f}%",
                    "Trades": v.get("trade_count", 0),
                }
                for k, v in sorted(regime_perf.items())
            ])
            st.dataframe(regime_df, use_container_width=True)
        else:
            st.info("No regime performance data available yet.")

    elif page == "Trade History":
        st.title("Trade History")
        if all_trades:
            df = pd.DataFrame(all_trades)
            df["return_pct"] = (df["return"] * 100).round(3)
            df["drawdown_pct"] = (df["drawdown"] * 100).round(3)
            display_cols = [
                "timestamp", "asset", "entry_price", "exit_price",
                "return_pct", "pnl", "drawdown_pct", "hold_time",
                "regime", "strategy_version", "exit_reason"
            ]
            existing = [c for c in display_cols if c in df.columns]
            st.dataframe(df[existing].tail(100), use_container_width=True)
            st.caption(f"Showing last 100 of {len(all_trades)} trades")
        else:
            st.info("No trades recorded yet. Run `python run.py` to start paper trading.")

    elif page == "Reflection History":
        st.title("Reflection & Optimization History")
        if reflections:
            for i, r in enumerate(reversed(reflections), 1):
                with st.expander(
                    f"Reflection #{len(reflections) - i + 1} — "
                    f"{r.get('timestamp', '')[:19]} | "
                    f"{r.get('changed_variable', '?')}: {r.get('old_value', '?')} → {r.get('new_value', '?')}"
                ):
                    st.markdown(f"**Variable Changed:** `{r.get('changed_variable', '')}`")
                    st.markdown(f"**Old Value:** {r.get('old_value', '')}")
                    st.markdown(f"**New Value:** {r.get('new_value', '')}")
                    st.markdown(f"**Reason:** {r.get('reason', '')}")
                    st.markdown(f"**Expected Effect:** {r.get('expected_effect', '')}")
                    st.markdown(f"**New Version:** v{r.get('version', '?')}")
        else:
            st.info("No reflections yet. Run enough trades to trigger the reflection cycle.")

    elif page == "Strategy Versions":
        st.title("Strategy Version History")
        if versions:
            for v in reversed(versions):
                with st.expander(f"Version v{v.get('version', '?')} — {v.get('_file', '')}"):
                    display = {k: val for k, val in v.items() if not k.startswith("_")}
                    st.json(display)
        else:
            st.info("No version history yet. Strategy modifications will appear here.")
        st.markdown("---")
        st.subheader("Current Active Strategy")
        st.json(strategy)


if __name__ == "__main__":
    main()
