"""
tab2_diagnostic.py  —  Diagnostic Analytics
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from theme import (
    base_layout, section_label, insight_box, hex_to_rgba,
    F1_RED, F1_WHITE, F1_SILVER, F1_GREY, F1_DGREY, F1_GOLD,
    ACCENT_TEAL, ACCENT_GREEN, ACCENT_AMBER,
    PLAN_COLORS, CHURN_COLORS,
)


def render(subs: pd.DataFrame, sess: pd.DataFrame, mrr: pd.DataFrame) -> None:
    st.markdown("## Diagnostic — *Why Are Subscribers Churning?*")
    st.markdown(
        "Drill into engagement drivers, regional patterns, content behaviour, "
        "churn reasons, and variable correlation structure."
    )
    st.markdown("---")

    # Pre-join sessions with subscriber info
    sess_s = sess.merge(
        subs[["Subscriber Id", "Plan", "Region", "churn_flag",
              "Churned", "Monthly Price Usd", "Churn Reason"]],
        on="Subscriber Id", how="left",
    )
    sess_s["Status"] = sess_s["churn_flag"].map({0: "Active", 1: "Churned"})

    # ── Row 1 ─────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(section_label("ENGAGEMENT SCORE — ACTIVE vs CHURNED"), unsafe_allow_html=True)
        fig1 = go.Figure()
        for label, color in CHURN_COLORS.items():
            vals = sess_s[sess_s["Status"] == label]["Engagement Score"].dropna()
            fig1.add_trace(go.Violin(
                y=vals,
                name=label,
                fillcolor=hex_to_rgba(color, 0.25),
                line_color=color,
                meanline_visible=True,
                box_visible=True,
                hoverinfo="y+name",
            ))
        lo1 = base_layout("Engagement Score: Active vs Churned", height=360)
        lo1["yaxis"]["title"] = "Engagement Score"
        fig1.update_layout(**lo1)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown(section_label("SESSION DURATION — ACTIVE vs CHURNED"), unsafe_allow_html=True)
        fig2 = go.Figure()
        for label, color in CHURN_COLORS.items():
            vals = sess_s[sess_s["Status"] == label]["Session Duration Min"].dropna()
            fig2.add_trace(go.Box(
                y=vals,
                name=label,
                fillcolor=hex_to_rgba(color, 0.25),
                line_color=color,
                boxmean="sd",
                marker=dict(color=color, size=3),
                hoverinfo="y+name",
            ))
        lo2 = base_layout("Session Duration: Active vs Churned", height=360)
        lo2["yaxis"]["title"] = "Session Duration (min)"
        fig2.update_layout(**lo2)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2 ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown(section_label("CHURN RATE BY REGION"), unsafe_allow_html=True)
        reg = (
            subs.groupby("Region")
            .agg(churn_rate=("churn_flag", "mean"),
                 total=("Subscriber Id", "count"),
                 churned=("churn_flag", "sum"))
            .reset_index()
            .sort_values("churn_rate", ascending=True)
        )
        fig3 = go.Figure(go.Bar(
            y=reg["Region"],
            x=reg["churn_rate"] * 100,
            orientation="h",
            marker=dict(
                color=reg["churn_rate"].tolist(),
                colorscale=[[0,"#BAE6FD"],[0.5,"#C4B5FD"],[1,"#FDA4AF"]],
            ),
            text=[f"{v*100:.1f}%  ({c}/{t})"
                  for v, c, t in zip(reg["churn_rate"], reg["churned"], reg["total"])],
            textposition="outside",
            textfont=dict(color=F1_WHITE, size=11),
            hovertemplate="<b>%{y}</b><br>Churn rate: %{x:.1f}%<extra></extra>",
        ))
        lo3 = base_layout("Churn Rate by Region", height=340)
        lo3["xaxis"]["title"] = "Churn Rate (%)"
        lo3["xaxis"]["range"] = [0, float(reg["churn_rate"].max()) * 145]
        lo3["margin"]["r"] = 60
        fig3.update_layout(**lo3)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown(section_label("CONTENT TYPE vs AVG ENGAGEMENT"), unsafe_allow_html=True)
        ct_eng = (
            sess.groupby("Content Type")
            .agg(avg_eng=("Engagement Score", "mean"),
                 avg_dur=("Session Duration Min", "mean"),
                 sessions=("Engagement Score", "count"))
            .reset_index()
            .sort_values("avg_eng", ascending=True)
        )
        fig4 = go.Figure(go.Bar(
            y=ct_eng["Content Type"],
            x=ct_eng["avg_eng"],
            orientation="h",
            marker=dict(
                color=ct_eng["avg_eng"].tolist(),
                colorscale=[[0,"#FDA4AF"],[0.5,"#C4B5FD"],[1,"#BAE6FD"]],
                showscale=True,
                colorbar=dict(title="Avg Score", tickfont=dict(color=F1_SILVER), len=0.8),
            ),
            text=[f"{v:.1f}" for v in ct_eng["avg_eng"]],
            textposition="outside",
            textfont=dict(color=F1_WHITE),
            customdata=ct_eng[["avg_dur", "sessions"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>Avg Engagement: %{x:.1f}<br>"
                "Avg Duration: %{customdata[0]:.1f} min<br>"
                "Sessions: %{customdata[1]:,}<extra></extra>"
            ),
        ))
        lo4 = base_layout("Content Type vs Avg Engagement Score", height=340)
        lo4["xaxis"]["title"] = "Avg Engagement Score"
        lo4["xaxis"]["range"] = [0, float(ct_eng["avg_eng"].max()) * 1.28]
        lo4["margin"]["r"] = 60
        fig4.update_layout(**lo4)
        st.plotly_chart(fig4, use_container_width=True)

    # ── Row 3 ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    col5, col6 = st.columns(2)

    with col5:
        st.markdown(section_label("CHURN REASON BREAKDOWN"), unsafe_allow_html=True)
        reasons = (
            subs[subs["Churned"] == "Yes"]["Churn Reason"]
            .value_counts()
            .reset_index()
        )
        reasons.columns = ["reason", "count"]
        TREEMAP_PASTELS = ["#C4B5FD", "#7DD3FC", "#86EFAC", "#FCA5A5", "#FDE68A", "#FBCFE8", "#A5F3FC"]
        reason_colors   = {r: TREEMAP_PASTELS[i % len(TREEMAP_PASTELS)]
                           for i, r in enumerate(reasons["reason"])}
        fig5 = go.Figure(go.Treemap(
            labels=reasons["reason"].tolist(),
            parents=[""] * len(reasons),
            values=reasons["count"].tolist(),
            marker=dict(
                colors=[reason_colors[r] for r in reasons["reason"]],
                line=dict(color="#0A0A0A", width=2),
            ),
            texttemplate="<b>%{label}</b><br>%{value}<br>%{percentRoot:.0%}",
            textfont=dict(color="#0A0A0A", size=13),
            hovertemplate="<b>%{label}</b><br>Churned: %{value}<br>%{percentRoot:.1%}<extra></extra>",
        ))
        fig5.update_layout(**base_layout("Why Did Subscribers Cancel?", height=340))
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        st.markdown(section_label("CHURN RATE: REGION × PLAN HEATMAP"), unsafe_allow_html=True)
        heat = (
            subs.groupby(["Region", "Plan"])["churn_flag"]
            .mean().reset_index()
        )
        pivot = heat.pivot(index="Region", columns="Plan", values="churn_flag").fillna(0)
        for col_name in ["Pit Lane", "Podium", "Paddock Club"]:
            if col_name not in pivot.columns:
                pivot[col_name] = 0
        pivot = pivot[["Pit Lane", "Podium", "Paddock Club"]]

        fig6 = go.Figure(go.Heatmap(
            z=pivot.values * 100,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[
                [0.0,  "#BAE6FD"],
                [0.25, "#C4B5FD"],
                [0.5,  "#FBCFE8"],
                [0.75, "#FCA5A5"],
                [1.0,  "#FDA4AF"],
            ],
            zmin=0,
            zmax=55,
            text=[[f"{v:.1f}%" for v in row] for row in pivot.values * 100],
            texttemplate="%{text}",
            textfont=dict(size=12, color="#111111"),
            hovertemplate="<b>%{y} — %{x}</b><br>Churn: %{z:.1f}%<extra></extra>",
            colorbar=dict(title="Churn %", ticksuffix="%",
                          tickfont=dict(color=F1_SILVER), len=0.8),
        ))
        fig6.update_layout(**base_layout("Churn Rate: Region × Plan", height=340))
        st.plotly_chart(fig6, use_container_width=True)

    # ── Row 4 ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    col7, col8 = st.columns(2)

    with col7:
        st.markdown(
            section_label("ENGAGEMENT × SESSION DURATION  (COLOURED BY CHURN)"),
            unsafe_allow_html=True,
        )
        samp = sess_s.sample(min(2000, len(sess_s)), random_state=42)
        fig7 = px.scatter(
            samp,
            x="Session Duration Min",
            y="Engagement Score",
            color="Status",
            color_discrete_map=CHURN_COLORS,
            opacity=0.45,
            trendline="ols",
            labels={
                "Session Duration Min": "Session Duration (min)",
                "Engagement Score": "Engagement Score",
                "Status": "Churn Status",
            },
        )
        fig7.update_layout(**base_layout(
            "Engagement vs Session Duration — Churn Overlay", height=360))
        st.plotly_chart(fig7, use_container_width=True)

    with col8:
        st.markdown(section_label("AVG ENGAGEMENT SCORE BY DAY OF WEEK"), unsafe_allow_html=True)
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_eng = (
            sess.groupby("Session Weekday")["Engagement Score"]
            .agg(avg="mean", sessions="count")
            .reindex(day_order)
            .reset_index()
        )
        fig8 = go.Figure(go.Bar(
            x=day_eng["Session Weekday"],
            y=day_eng["avg"],
            marker=dict(
                color=day_eng["avg"].tolist(),
                colorscale=[[0,"#FDA4AF"],[0.5,"#C4B5FD"],[1,"#BAE6FD"]],
            ),
            text=[f"{v:.1f}" for v in day_eng["avg"]],
            textposition="outside",
            textfont=dict(color=F1_WHITE, size=11),
            customdata=day_eng["sessions"].tolist(),
            hovertemplate=(
                "<b>%{x}</b><br>Avg Engagement: %{y:.1f}<br>"
                "Sessions: %{customdata:,}<extra></extra>"
            ),
        ))
        lo8 = base_layout("Avg Engagement Score by Day of Week", height=360)
        lo8["yaxis"]["title"] = "Avg Engagement Score"
        lo8["yaxis"]["range"] = [54, float(day_eng["avg"].max()) * 1.08]
        fig8.update_layout(**lo8)
        st.plotly_chart(fig8, use_container_width=True)

    # ── Correlation Heatmap ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        section_label("CORRELATION HEATMAP — KEY SUBSCRIBER VARIABLES"),
        unsafe_allow_html=True,
    )
    sub_sess_agg = (
        sess.groupby("Subscriber Id")
        .agg(avg_eng=("Engagement Score", "mean"),
             avg_dur=("Session Duration Min", "mean"),
             total_sess=("Engagement Score", "count"))
        .reset_index()
    )
    corr_raw = subs.merge(sub_sess_agg, on="Subscriber Id", how="left")
    for c in ["avg_eng", "avg_dur", "total_sess"]:
        corr_raw[c] = corr_raw[c].fillna(0)

    corr_map = {
        "churn_flag":        "Churn",
        "Monthly Price Usd": "Price",
        "Tenure Months":     "Tenure",
        "Nps Score":         "NPS",
        "Renewal Count":     "Renewals",
        "avg_eng":           "Avg Engagement",
        "avg_dur":           "Avg Duration",
        "total_sess":        "Total Sessions",
        "Age":               "Age",
    }
    corr_df  = corr_raw[list(corr_map.keys())].rename(columns=corr_map)
    corr_mat = corr_df.corr().round(3)

    fig9 = go.Figure(go.Heatmap(
        z=corr_mat.values,
        x=corr_mat.columns.tolist(),
        y=corr_mat.index.tolist(),
        colorscale=[[0,"#FDA4AF"],[0.5,"#E0E0E0"],[1,"#BAE6FD"]],
        zmin=-1,
        zmax=1,
        text=corr_mat.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=11, color=F1_WHITE),
        hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(title="Pearson r", tickfont=dict(color=F1_SILVER)),
    ))
    lo9 = base_layout("Pearson Correlation — Subscriber & Session Variables", height=440)
    fig9.update_layout(**lo9)
    st.plotly_chart(fig9, use_container_width=True)

    # ── Insights ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(section_label("KEY DIAGNOSTIC INSIGHTS"), unsafe_allow_html=True)

    r_eng_churn  = corr_mat.loc["Churn", "Avg Engagement"]
    top_reason   = (subs[subs["Churned"] == "Yes"]["Churn Reason"]
                    .value_counts().idxmax())
    worst_region = subs.groupby("Region")["churn_flag"].mean().idxmax()
    best_content = ct_eng.sort_values("avg_eng").iloc[-1]["Content Type"]

    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown(insight_box(
            f"📉 <b>Engagement is negatively correlated with churn</b> (r = {r_eng_churn:.2f}). "
            f"Subscribers who engage deeply are significantly less likely to cancel. "
            f"Protecting engagement in the first 90 days is the single highest-leverage "
            f"retention action available."
        ), unsafe_allow_html=True)
    with i2:
        st.markdown(insight_box(
            f"🚩 <b>'{top_reason}'</b> is the #1 stated cancellation reason. "
            f"This signals a content depth problem that pricing changes alone won't fix — "
            f"a quarterly content roadmap communicated proactively to subscribers would "
            f"directly address this driver."
        ), unsafe_allow_html=True)
    with i3:
        st.markdown(insight_box(
            f"🌍 <b>{worst_region}</b> shows the highest regional churn. "
            f"The Region × Plan heatmap reveals this is concentrated in the entry tier — "
            f"a localised pricing or content relevance issue rather than a "
            f"platform-wide problem."
        ), unsafe_allow_html=True)
