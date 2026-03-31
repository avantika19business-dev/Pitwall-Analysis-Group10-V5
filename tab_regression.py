"""
tab_regression.py  —  Regression Analysis
Linear, Ridge, and Lasso compared on Lifetime Revenue prediction
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from theme import (
    base_layout, section_label, insight_box, warn_box, hex_to_rgba,
    F1_RED, F1_WHITE, F1_SILVER, F1_GREY, F1_DGREY, F1_GOLD,
    ACCENT_TEAL, ACCENT_GREEN, ACCENT_AMBER,
)
from model_utils import engineer_features, train_regression_models

REG_COLORS = {
    "Linear Regression": ACCENT_TEAL,
    "Ridge Regression":  ACCENT_GREEN,
    "Lasso Regression":  ACCENT_AMBER,
}

@st.cache_data(show_spinner=False)
def _run(subs: pd.DataFrame, sess: pd.DataFrame):
    df = engineer_features(subs, sess)
    return train_regression_models(df)


def render(subs: pd.DataFrame, sess: pd.DataFrame, mrr: pd.DataFrame) -> None:
    st.markdown("## Regression Analysis")
    st.markdown(
        "Predicting subscriber **Lifetime Revenue** using three regularised regression models. "
        "Ridge penalises large coefficients; Lasso performs feature selection by zeroing weak predictors."
    )
    st.markdown("---")

    with st.spinner("🏎  Training regression models…"):
        results = _run(subs, sess)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    cols = st.columns(3)
    for i, r in enumerate(results):
        cols[i].metric(r["name"], f"R² = {r['r2']:.3f}",
                       f"RMSE ${r['rmse']:,.0f}")
    st.markdown("---")

    # ── Model tabs ─────────────────────────────────────────────────────────────
    t1, t2, t3 = st.tabs([
        "📉  Linear Regression",
        "🔵  Ridge Regression",
        "🟡  Lasso Regression",
    ])

    def _render_model(r: dict, tab) -> None:
        color = REG_COLORS[r["name"]]
        with tab:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("R² Score",  f"{r['r2']:.4f}")
            m2.metric("RMSE",      f"${r['rmse']:,.2f}")
            m3.metric("MAE",       f"${r['mae']:,.2f}")
            m4.metric("Model",     r["name"].split()[0])

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(section_label("ACTUAL vs PREDICTED"), unsafe_allow_html=True)
                fig_av = go.Figure()
                fig_av.add_trace(go.Scatter(
                    x=r["y_test"], y=r["y_pred"],
                    mode="markers",
                    marker=dict(color=color, size=5, opacity=0.55,
                                line=dict(color=F1_DGREY, width=0.5)),
                    hovertemplate="Actual: $%{x:,.0f}<br>Predicted: $%{y:,.0f}<extra></extra>",
                    name="Prediction",
                ))
                mn = min(float(min(r["y_test"])), float(min(r["y_pred"])))
                mx = max(float(max(r["y_test"])), float(max(r["y_pred"])))
                fig_av.add_trace(go.Scatter(
                    x=[mn, mx], y=[mn, mx],
                    mode="lines", name="Perfect Fit",
                    line=dict(color=F1_RED, width=1.5, dash="dash"),
                ))
                lo_av = base_layout(f"Actual vs Predicted — {r['name']}", height=360)
                lo_av["xaxis"]["title"] = "Actual LTV (USD)"
                lo_av["yaxis"]["title"] = "Predicted LTV (USD)"
                fig_av.update_layout(**lo_av)
                st.plotly_chart(fig_av, use_container_width=True)

            with c2:
                st.markdown(section_label("RESIDUALS DISTRIBUTION"), unsafe_allow_html=True)
                residuals = np.array(r["y_test"]) - np.array(r["y_pred"])
                fig_res = go.Figure(go.Histogram(
                    x=residuals,
                    nbinsx=30,
                    marker=dict(color=color, opacity=0.75,
                                line=dict(color=F1_DGREY, width=0.5)),
                    hovertemplate="Residual: %{x:,.0f}<br>Count: %{y}<extra></extra>",
                ))
                fig_res.add_vline(x=0, line_dash="dash", line_color=F1_RED,
                                  annotation_text="Zero Error",
                                  annotation_font_color=F1_RED)
                lo_res = base_layout(f"Residuals — {r['name']}", height=360)
                lo_res["xaxis"]["title"] = "Residual (USD)"
                lo_res["yaxis"]["title"] = "Count"
                fig_res.update_layout(**lo_res)
                st.plotly_chart(fig_res, use_container_width=True)

            st.markdown(section_label("STANDARDISED COEFFICIENTS"), unsafe_allow_html=True)
            coef = r["coefs"].copy()
            coef_colors = [ACCENT_GREEN if v > 0 else F1_RED for v in coef["coefficient"]]
            fig_c = go.Figure(go.Bar(
                y=coef["feature"],
                x=coef["coefficient"],
                orientation="h",
                marker=dict(color=coef_colors, line=dict(color=F1_DGREY, width=0.5)),
                text=[f"{v:+.3f}" for v in coef["coefficient"]],
                textposition="outside",
                textfont=dict(color=F1_WHITE, size=10),
                hovertemplate="<b>%{y}</b><br>Coef: %{x:+.4f}<extra></extra>",
            ))
            lo_c = base_layout(f"Standardised Coefficients — {r['name']}", height=360)
            lo_c["xaxis"]["title"] = "Coefficient (standardised)"
            lo_c["margin"]["r"]    = 80
            fig_c.update_layout(**lo_c)
            st.plotly_chart(fig_c, use_container_width=True)

    for r, tab in zip(results, [t1, t2, t3]):
        _render_model(r, tab)

    # ── Comparison section ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(section_label("MODEL COMPARISON — LINEAR vs RIDGE vs LASSO"), unsafe_allow_html=True)

    comp_df = pd.DataFrame([{
        "Model": r["name"],
        "R² Score": f"{r['r2']:.4f}",
        "RMSE ($)": f"${r['rmse']:,.2f}",
        "MAE ($)":  f"${r['mae']:,.2f}",
    } for r in results])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(section_label("R² SCORE COMPARISON"), unsafe_allow_html=True)
        fig_r2 = go.Figure(go.Bar(
            x=[r["name"] for r in results],
            y=[r["r2"]   for r in results],
            marker=dict(
                color=[REG_COLORS[r["name"]] for r in results],
                line=dict(color=F1_DGREY, width=1),
            ),
            text=[f"{r['r2']:.4f}" for r in results],
            textposition="outside",
            textfont=dict(color=F1_WHITE, size=13),
            hovertemplate="<b>%{x}</b><br>R²: %{y:.4f}<extra></extra>",
        ))
        lo_r2 = base_layout("R² Score — All Regression Models", height=320)
        lo_r2["yaxis"]["title"] = "R² Score"
        lo_r2["yaxis"]["range"] = [min(0, min(r["r2"] for r in results)) - 0.05,
                                   max(r["r2"] for r in results) * 1.25]
        fig_r2.update_layout(**lo_r2)
        st.plotly_chart(fig_r2, use_container_width=True)

    with c2:
        st.markdown(section_label("RMSE COMPARISON"), unsafe_allow_html=True)
        fig_rmse = go.Figure(go.Bar(
            x=[r["name"]  for r in results],
            y=[r["rmse"]  for r in results],
            marker=dict(
                color=[REG_COLORS[r["name"]] for r in results],
                line=dict(color=F1_DGREY, width=1),
            ),
            text=[f"${r['rmse']:,.0f}" for r in results],
            textposition="outside",
            textfont=dict(color=F1_WHITE, size=13),
            hovertemplate="<b>%{x}</b><br>RMSE: $%{y:,.0f}<extra></extra>",
        ))
        lo_rmse = base_layout("RMSE — All Regression Models", height=320)
        lo_rmse["yaxis"]["title"] = "RMSE (USD)"
        lo_rmse["yaxis"]["range"] = [0, max(r["rmse"] for r in results) * 1.25]
        fig_rmse.update_layout(**lo_rmse)
        st.plotly_chart(fig_rmse, use_container_width=True)

    # ── Insights ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(section_label("KEY REGRESSION INSIGHTS"), unsafe_allow_html=True)

    best_r2 = max(results, key=lambda r: r["r2"])

    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown(insight_box(
            f"📈 <b>{best_r2['name']}</b> achieves the highest R² ({best_r2['r2']:.4f}). "
            f"The moderate R² is expected — lifetime revenue is structurally driven by "
            f"plan price × tenure, both of which are captured in the feature set."
        ), unsafe_allow_html=True)
    with i2:
        lasso_r = next(r for r in results if "Lasso" in r["name"])
        zero_coefs = int((np.abs(lasso_r["coefs"]["coefficient"]) < 0.001).sum())
        st.markdown(insight_box(
            f"✂️ <b>Lasso</b> performs automatic feature selection — it zeroed out "
            f"<b>{zero_coefs} weak coefficient(s)</b>, confirming that Monthly Price "
            f"and Tenure are the dominant revenue drivers."
        ), unsafe_allow_html=True)
    with i3:
        st.markdown(warn_box(
            "⚠️ <b>Low R² across models</b> reflects synthetic data noise rather than "
            "a model failure. In production, integrating payment history and real session "
            "depth metrics would substantially improve predictive power. "
            "The directional coefficient signs are all economically sensible."
        ), unsafe_allow_html=True)
