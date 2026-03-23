"""
app.py — Streamlit UI only.

All computation is in pipeline.py.
This file only renders what pipeline.run_pipeline() returns.
~120 lines vs the previous ~560 lines.
"""

import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import pandas as pd

from pipeline import INITIAL_CASH, run_pipeline

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Stock Trend Prediction",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Stock Market Trend Prediction")
st.caption("LSTM+XGBoost · TFT · GNN · Technical · Fundamental · Sentiment · Regime")

# ── Sidebar settings ──────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    use_tft   = st.toggle("Temporal Fusion Transformer", value=False)
    use_gnn   = st.toggle("Graph Neural Network",        value=False)
    use_macro = st.toggle("India VIX + USD/INR",         value=True)
    st.divider()
    st.caption("Base: LSTM+XGBoost · Walk-forward CV · ATR stop-loss")

# ── Stock selector ────────────────────────────────────────
STOCKS = {
    # Large-caps
    "TCS":            "TCS.NS",      "Reliance":      "RELIANCE.NS",
    "Infosys":        "INFY.NS",     "HDFC Bank":     "HDFCBANK.NS",
    "ICICI Bank":     "ICICIBANK.NS","L&T":           "LT.NS",
    "ITC":            "ITC.NS",      "SBI":           "SBIN.NS",
    "Bharti Airtel":  "BHARTIARTL.NS","HCL Tech":     "HCLTECH.NS",
    # Midcap IT
    "Persistent Sys": "PERSISTENT.NS","Coforge":      "COFORGE.NS",
    "KPIT Tech":      "KPITTECH.NS", "Mphasis":       "MPHASIS.NS",
    # Midcap Banking
    "Federal Bank":   "FEDERALBNK.NS","IDFC First":   "IDFCFIRSTB.NS",
    "Bandhan Bank":   "BANDHANBNK.NS","Cholamandalam":"CHOLAFIN.NS",
    # Midcap FMCG
    "Tata Consumer":  "TATACONSUM.NS","Godrej Consumer":"GODREJCP.NS",
    "Marico":         "MARICO.NS",
    # Midcap Pharma
    "Alkem Labs":     "ALKEM.NS",    "Ipca Labs":     "IPCALAB.NS",
    "Torrent Pharma": "TORNTPHARM.NS",
    # Midcap Auto
    "Ashok Leyland":  "ASHOKLEY.NS", "Escorts Kubota":"ESCORTS.NS",
    "Balkrishna Ind": "BALKRISIND.NS",
}

selected_name = st.selectbox("Select stock", list(STOCKS.keys()))
ticker        = STOCKS[selected_name]


# ── Pipeline (cached per ticker + settings) ───────────────
@st.cache_data(ttl=3600, show_spinner="Running full analysis…")
def load_all_stocks_for_gnn():
    dfs = {}
    for t in STOCKS.values():
        try:
            df = yf.download(t, period="2y", interval="1d",
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                dfs[t] = df
        except Exception:
            pass
    return dfs


with st.spinner(f"Analysing {selected_name}…"):
    all_dfs = load_all_stocks_for_gnn() if use_gnn else None
    r = run_pipeline(ticker, use_tft=use_tft, use_gnn=use_gnn,
                     use_macro=use_macro, all_dfs=all_dfs)

if r.error:
    st.error(f"Pipeline error: {r.error}")
    st.stop()



# ── Unpack for convenience ────────────────────────────────
final   = r.final
tech    = r.tech
fund    = r.fund
regime  = r.regime
nse     = r.nse

# ══════════════════════════════════════════════════════════
# UI — purely rendering from here down
# ══════════════════════════════════════════════════════════

# ── Banners ───────────────────────────────────────────────
reg_color = "🟢" if regime["regime"] == "BULL" else ("🔴" if regime["regime"] == "BEAR" else "⚪")
st.info(f"{reg_color} **Market regime:** {regime['signal']}")

if final.get("regime_override"):
    st.warning(f"⚡ {final['regime_override']}")
if final.get("earnings_risk") == "HIGH":
    st.warning(f"⚠️ Earnings within 14 days — BUY blocked to reduce event risk")

sig_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(final["signal"], "⚪")
st.markdown(f"## {sig_color} Final signal: **{final['signal']}**  —  confidence {final['confidence']:.1f}%")

# ── Model signals ─────────────────────────────────────────
st.subheader("🤖 Model signals")
mc = st.columns(4)
mc[0].metric("LSTM+XGB",  r.ml_signal,  f"{r.ml_confidence*100:.1f}%")
mc[1].metric("Technical", tech["signal"])
mc[2].metric("TFT",  r.tft_signal, f"{r.tft_confidence*100:.1f}%" if use_tft else "off")
mc[3].metric("GNN",  r.gnn_signal, f"{r.gnn_confidence*100:.1f}%" if use_gnn else "off")

# ── Macro indicators ──────────────────────────────────────
if use_macro and (tech.get("vix") or tech.get("usdinr")):
    st.subheader("🌐 Macro indicators")
    m1, m2, m3 = st.columns(3)
    if tech.get("vix"):
        vix = tech["vix"]
        m1.metric("India VIX", f"{vix:.1f}",
                  "High fear" if vix > 20 else ("Low fear" if vix < 13 else "Neutral"))
    if tech.get("usdinr"):
        usd = tech["usdinr"]
        m2.metric("USD/INR", f"{usd:.2f}",
                  "Weak rupee" if usd > 84 else ("Strong rupee" if usd < 82 else "Neutral"))
    m3.metric("Nifty50", f"₹{regime['last_close']:,.0f}",
              f"MA50 vs MA200: {regime['gap_pct']:+.1f}%")

# ── NSE signals ───────────────────────────────────────────
pcr   = nse.get("pcr")
e_risk = nse.get("earnings_risk", "UNKNOWN")
s_sig  = nse.get("sector_signal", "UNKNOWN")
d_to_e = nse.get("days_to_earnings")
if any(v is not None for v in [pcr, e_risk, s_sig]):
    st.subheader("📋 NSE signals")
    n1, n2, n3 = st.columns(3)
    if pcr:
        n1.metric("Nifty PCR", f"{pcr:.2f}", nse.get("pcr_signal", ""))
    n2.metric("Sector momentum", s_sig,
              f"5d: {nse.get('sector_5d',0):+.1f}%  21d: {nse.get('sector_21d',0):+.1f}%")
    n3.metric("Earnings risk", e_risk,
              f"{d_to_e} days" if d_to_e else "Unknown")

# ── Price chart + component scores ────────────────────────
col_chart, col_scores = st.columns([2, 1])
with col_chart:
    st.subheader("Price chart + signals")
    buy_idx  = r.df_bt.index[r.signals == "BUY"]
    sell_idx = r.df_bt.index[r.signals == "SELL"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r.df_bt.index, y=r.df_bt["Close"],
                             mode="lines", name="Close", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=buy_idx, y=r.df_bt.loc[buy_idx, "Close"],
                             mode="markers", name="BUY",
                             marker=dict(symbol="triangle-up", size=10, color="lime")))
    fig.add_trace(go.Scatter(x=sell_idx, y=r.df_bt.loc[sell_idx, "Close"],
                             mode="markers", name="SELL",
                             marker=dict(symbol="triangle-down", size=10, color="red")))
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=0, b=0),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  x=1, xanchor="right"))
    st.plotly_chart(fig, width="stretch")

with col_scores:
    st.subheader("Component scores")
    cs = final["component_scores"]
    for label, key in [("Technical","technical"), ("Fundamental","fundamental"),
                       ("Sentiment","sentiment"), ("ML","ml")]:
        if cs.get(key) is not None:
            st.metric(label, f"{cs[key]*100:.1f}%")
    if cs.get("tft") is not None:
        st.metric("TFT", f"{cs['tft']*100:.1f}%")
    if cs.get("gnn") is not None:
        st.metric("GNN", f"{cs['gnn']*100:.1f}%")

# ── Analysis tabs ─────────────────────────────────────────
tab_tech, tab_fund, tab_sent, tab_ml, tab_tft, tab_gnn, tab_bt = st.tabs([
    "⚙️ Technical", "📊 Fundamental", "📰 Sentiment",
    "🤖 LSTM+XGB",  "🔮 TFT",         "🕸️ GNN", "📈 Backtest",
])

with tab_tech:
    st.write(f"**Signal:** {tech['signal']}  |  **Score:** {tech['score']:.2f}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### ✅ Bullish")
        for x in tech["buy_reasons"]  or ["—"]: st.write("✔", x)
    with c2:
        st.markdown("##### ⚠️ Risks")
        for x in tech["sell_reasons"] or ["—"]: st.write("⚠", x)

with tab_fund:
    scored = fund.get("fields_scored", "?")
    st.write(f"**Signal:** {fund['signal']}  |  **Score:** {fund['score']:.2f}  |  **Fields scored:** {scored}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### ✅ Bullish")
        for x in fund["buy_reasons"]  or ["—"]: st.write("✔", x)
    with c2:
        st.markdown("##### ⚠️ Risks")
        for x in fund["sell_reasons"] or ["—"]: st.write("⚠", x)
    with st.expander("Raw fundamental data"):
        for k, v in r.fundamental_data.items():
            st.write(f"**{k}:** {round(v,4) if v is not None else 'N/A'}")

with tab_sent:
    sent = r.sentiment
    st.write(f"**Signal:** {sent['signal']}  |  **Score:** {sent['score']:.4f}  |  **Engine:** {sent.get('engine','?')}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🟢 Positive")
        for x in sent["positive_news"] or ["—"]: st.write("✔", x)
    with c2:
        st.markdown("##### 🔴 Negative")
        for x in sent["negative_news"] or ["—"]: st.write("⚠", x)

with tab_ml:
    gate_label = "PASSED" if r.ml_confidence >= 0.55 else "BLOCKED (<55%)"
    st.write(f"**Signal:** {r.ml_signal}  |  **Confidence:** {r.ml_confidence*100:.1f}%  |  **Gate:** {gate_label}")
    if r.ml_result.is_valid:
        m = r.ml_result.metrics
        st.markdown("##### Walk-forward CV")
        wf = st.columns(4)
        wf[0].metric("WF Accuracy",  f"{m['wf_accuracy']}%")
        wf[1].metric("WF Precision", f"{m['wf_precision']}%")
        wf[2].metric("WF Recall",    f"{m['wf_recall']}%")
        wf[3].metric("WF F1",        f"{m['wf_f1']}%")
        st.markdown("##### Holdout test")
        ho = st.columns(4)
        ho[0].metric("Accuracy",  f"{m['accuracy']}%")
        ho[1].metric("Precision", f"{m['precision']}%")
        ho[2].metric("Recall",    f"{m['recall']}%")
        ho[3].metric("F1",        f"{m['f1']}%")
        st.caption(
            f"Trained {m['n_train']} days · tested {m['n_test']} days · "
            f"{m['wf_folds']} WF folds · {m.get('n_features',27)} features · "
            f"pos_weight={m.get('pos_weight','?')} · "
            f"Strategy return: {m['cumulative_strategy_return']:.2f}%"
        )
    else:
        st.warning("Model not trained (insufficient data).")

with tab_tft:
    if not use_tft:
        st.info("Enable TFT in the sidebar.")
    elif r.tft_result and r.tft_result.is_valid:
        st.write(f"**Signal:** {r.tft_signal}  |  **Confidence:** {r.tft_confidence*100:.1f}%")
        m = r.tft_result.metrics
        cols = st.columns(2)
        cols[0].metric("Accuracy", f"{m['accuracy']}%")
        cols[1].metric("MAE",      f"{m['mae']:.6f}")
        if r.tft_result.feature_importance is not None:
            st.markdown("##### Feature importance")
            st.dataframe(r.tft_result.feature_importance, width="stretch")
    else:
        st.warning("TFT failed. Check pytorch-forecasting is installed.")

with tab_gnn:
    if not use_gnn:
        st.info("Enable GNN in the sidebar.")
    elif r.gnn_result and r.gnn_result.is_valid:
        st.write(f"**Signal:** {r.gnn_signal}  |  **Confidence:** {r.gnn_confidence*100:.1f}%")
        m = r.gnn_result.metrics
        cols = st.columns(3)
        cols[0].metric("Accuracy", f"{m['accuracy']}%")
        cols[1].metric("Stocks",   m['n_stocks'])
        cols[2].metric("Edges",    m['n_edges'])
        if r.gnn_result.edge_df is not None and not r.gnn_result.edge_df.empty:
            st.markdown("##### Correlation graph")
            st.dataframe(
                r.gnn_result.edge_df.sort_values("correlation", key=abs, ascending=False),
                width="stretch",
            )
    else:
        st.warning("GNN failed. Check torch-geometric is installed.")

with tab_bt:
    no_trades = r.trades.empty or (
        not r.trades.empty and
        r.trades["action"].str.contains("OPEN").all()
    )
    if no_trades:
        bull_days = int(r.hist_regime.sum()) if not r.hist_regime.empty else 0
        bear_days = len(r.df_short) - bull_days
        st.info(
            f"💰 **No trades executed** — signals blocked by confidence gate (55%) "
            f"and/or regime filter.  \n"
            f"📊 {bull_days} BULL days vs {bear_days} BEAR days over backtest period."
        )

    st.markdown("### Strategy vs Nifty50")
    if no_trades:
        bm_ret = r.benchmark.get("total_return_pct", 0)
        verdict_color = "🟢" if 0 > bm_ret else "🔴"
        st.write(f"{verdict_color} Strategy preserved capital (0%) vs Nifty50 {bm_ret:+.2f}%")
        cols = st.columns(2)
        cols[0].metric("Strategy (no trades)", "0.00%", "Capital preserved")
        cols[1].metric("Nifty50 buy-and-hold", f"{bm_ret:.2f}%")
    else:
        comp = r.comparison
        verdict_color = "🟢" if comp["verdict"] == "OUTPERFORMING" else (
                        "🔴" if comp["verdict"] == "UNDERPERFORMING" else "🟡")
        st.write(f"{verdict_color} **{comp['verdict']}** the benchmark")
        cols = st.columns(4)
        cols[0].metric("Strategy return", f"{comp['strategy_return']}%",
                       f"{comp['return_alpha']:+.2f}% vs Nifty50")
        cols[1].metric("Nifty50 return",  f"{comp['benchmark_return']}%")
        cols[2].metric("Strategy Sharpe", f"{comp['strategy_sharpe']}",
                       f"{comp['sharpe_difference']:+.2f} vs Nifty50")
        cols[3].metric("Nifty50 Sharpe",  f"{comp['benchmark_sharpe']}")
        st.markdown("### Performance metrics")
        mc2 = st.columns(len(r.metrics))
        for col, (k, v) in zip(mc2, r.metrics.items()):
            col.metric(k, v)

    st.markdown("### Portfolio equity curve vs Nifty50")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=r.df_bt.index, y=r.df_bt["Portfolio"],
                              mode="lines", fill="tozeroy", name="Strategy",
                              line=dict(color="#378ADD")))
    if not r.benchmark.get("series", pd.Series()).empty:
        bm_s = r.benchmark["series"].reindex(r.df_bt.index, method="ffill")
        if float(bm_s.iloc[0]) != 0:
            bm_s = bm_s * (INITIAL_CASH / float(bm_s.iloc[0]))
        fig2.add_trace(go.Scatter(x=bm_s.index, y=bm_s,
                                  mode="lines", name="Nifty50",
                                  line=dict(color="#888780", dash="dash")))
    fig2.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                   x=1, xanchor="right"))
    st.plotly_chart(fig2, width="stretch")

    st.markdown("### Trade log")
    if r.trades.empty:
        st.info("No trades executed.")
    else:
        st.dataframe(r.trades, width="stretch")
        sig_counts = r.signals.value_counts()
        st.caption("  ·  ".join(f"{k}: {v}" for k, v in sig_counts.items()))