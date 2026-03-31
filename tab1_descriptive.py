"""
tab1_descriptive.py  —  Descriptive Analytics
"""
from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from theme import (
    base_layout, section_label, insight_box, hex_to_rgba,
    F1_RED, F1_WHITE, F1_SILVER, F1_GREY, F1_DGREY, F1_GOLD,
    ACCENT_TEAL, ACCENT_GREEN, ACCENT_AMBER,
    PLAN_COLORS, CHANNEL_COLORS, NPS_COLORS,
)


def render(subs: pd.DataFrame, sess: pd.DataFrame, mrr: pd.DataFrame) -> None:
    st.markdown("## Descriptive — *Who Are Our Subscribers?*")
    st.markdown(
        "Platform-wide snapshot: subscriber base health, plan distribution, "
        "engagement patterns, device behaviour, and revenue trajectory."
    )
    st.markdown("---")

    # ── KPI Row ───────────────────────────────────────────────────────────────
    total_subs  = len(subs)
    active_subs = (subs["Churned"] == "No").sum()
    churn_rate  = subs["churn_flag"].mean() * 100
    avg_eng     = sess["Engagement Score"].mean()
    avg_dur     = sess["Session Duration Min"].mean()
    latest_mrr  = mrr.groupby("Month")["Mrr Usd"].sum().iloc[-1]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Subscribers",    f"{total_subs:,}")
    c2.metric("Active Subscribers",   f"{active_subs:,}", f"{active_subs/total_subs*100:.1f}% of base")
    c3.metric("Platform Churn Rate",  f"{churn_rate:.1f}%")
    c4.metric("Avg Engagement Score", f"{avg_eng:.1f} / 100")
    c5.metric("Avg Session Duration", f"{avg_dur:.1f} min")
    c6.metric("Latest Month MRR",     f"${latest_mrr:,.0f}")
    st.markdown("---")

    # ── Row 1 ─────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(section_label("RETENTION RATE BY PLAN TIER"), unsafe_allow_html=True)
        plan_stats = (
            subs.groupby("Plan")
            .agg(total=("Subscriber Id", "count"), churned=("churn_flag", "sum"))
            .reset_index()
        )
        plan_stats["retention_pct"] = (1 - plan_stats["churned"] / plan_stats["total"]) * 100
        plan_stats["churn_pct"]     = plan_stats["churned"] / plan_stats["total"] * 100
        _order = {"Pit Lane": 0, "Podium": 1, "Paddock Club": 2}
        plan_stats = plan_stats.sort_values("Plan", key=lambda s: s.map(_order))

        fig1 = go.Figure(go.Bar(
            x=plan_stats["Plan"],
            y=plan_stats["retention_pct"],
            marker=dict(
                color=[PLAN_COLORS.get(p, F1_SILVER) for p in plan_stats["Plan"]],
                line=dict(color=F1_DGREY, width=1),
            ),
            text=[f"{v:.1f}%" for v in plan_stats["retention_pct"]],
            textposition="outside",
            textfont=dict(color=F1_WHITE, size=13),
            customdata=plan_stats[["churned", "total", "churn_pct"]].values,
            hovertemplate=(
                "<b>%{x}</b><br>Retention: %{y:.1f}%<br>"
                "Churned: %{customdata[0]} / %{customdata[1]}<br>"
                "Churn rate: %{customdata[2]:.1f}%<extra></extra>"
            ),
        ))
        lo1 = base_layout("Subscriber Retention by Plan Tier", height=340)
        lo1["yaxis"]["range"] = [0, 115]
        lo1["yaxis"]["title"] = "Retention %"
        fig1.update_layout(**lo1)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown(section_label("CHURN RATE BY PRICING TIER"), unsafe_allow_html=True)
        price_churn = (
            subs.groupby("Plan")
            .agg(churn_rate=("churn_flag", "mean"),
                 price=("Monthly Price Usd", "first"),
                 count=("Subscriber Id", "count"))
            .reset_index()
            .sort_values("price")
        )
        fig2 = go.Figure(go.Bar(
            x=[f"${p:.2f}/mo  {pl}" for p, pl in
               zip(price_churn["price"], price_churn["Plan"])],
            y=price_churn["churn_rate"] * 100,
            marker=dict(
                color=price_churn["churn_rate"].tolist(),
                colorscale=[[0,"#BAE6FD"],[0.5,"#C4B5FD"],[1,"#FDA4AF"]],
                showscale=True,
                colorbar=dict(title="Churn Rate", tickformat=".0%",
                              tickfont=dict(color=F1_SILVER), len=0.8),
            ),
            text=[f"{v*100:.1f}%" for v in price_churn["churn_rate"]],
            textposition="outside",
            textfont=dict(color=F1_WHITE, size=13),
            customdata=price_churn["count"].tolist(),
            hovertemplate=(
                "<b>%{x}</b><br>Churn: %{y:.1f}%<br>"
                "Subscribers: %{customdata}<extra></extra>"
            ),
        ))
        lo2 = base_layout("Churn Rate by Pricing Tier", height=340)
        lo2["yaxis"]["title"] = "Churn Rate (%)"
        lo2["yaxis"]["range"] = [0, float(price_churn["churn_rate"].max()) * 145]
        fig2.update_layout(**lo2)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2 ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown(section_label("ENGAGEMENT SCORE DISTRIBUTION BY PLAN"), unsafe_allow_html=True)
        sess_plan = sess.merge(subs[["Subscriber Id", "Plan"]], on="Subscriber Id", how="left")
        fig3 = go.Figure()
        for plan, color in PLAN_COLORS.items():
            vals = sess_plan[sess_plan["Plan"] == plan]["Engagement Score"].dropna()
            fig3.add_trace(go.Violin(
                y=vals,
                name=plan,
                fillcolor=hex_to_rgba(color, 0.25),
                line_color=color,
                meanline_visible=True,
                box_visible=True,
                hoverinfo="y+name",
            ))
        lo3 = base_layout("Engagement Score Distribution by Plan", height=360)
        lo3["yaxis"]["title"] = "Engagement Score"
        fig3.update_layout(**lo3)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown(section_label("DEVICE USAGE BY PLAN"), unsafe_allow_html=True)
        dev_plan = (
            sess.merge(subs[["Subscriber Id", "Plan"]], on="Subscriber Id", how="left")
            .groupby(["Device", "Plan"]).size()
            .reset_index(name="sessions")
        )
        fig4 = px.bar(
            dev_plan, x="Device", y="sessions", color="Plan",
            color_discrete_map=PLAN_COLORS,
            barmode="stack",
            labels={"sessions": "Sessions", "Device": "Device Type"},
        )
        lo4 = base_layout("Device Usage — Sessions by Plan", height=360)
        lo4["yaxis"]["title"] = "Sessions"
        lo4["xaxis"]["title"] = "Device Type"
        fig4.update_layout(**lo4)
        fig4.update_traces(hovertemplate="<b>%{x}</b><br>%{y:,} sessions<extra></extra>")
        st.plotly_chart(fig4, use_container_width=True)

    # ── Row 3 ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    col5, col6 = st.columns(2)

    with col5:
        st.markdown(section_label("SESSION VOLUME BY CONTENT TYPE"), unsafe_allow_html=True)
        ct = sess["Content Type"].value_counts().reset_index()
        ct.columns = ["content", "count"]
        fig5 = go.Figure(go.Bar(
            y=ct["content"],
            x=ct["count"],
            orientation="h",
            marker=dict(
                color=ct["count"].tolist(),
                colorscale=[[0,"#BAE6FD"],[0.5,"#C4B5FD"],[1,"#FDA4AF"]],
            ),
            text=ct["count"].tolist(),
            textposition="outside",
            textfont=dict(color=F1_WHITE, size=11),
            hovertemplate="<b>%{y}</b><br>Sessions: %{x:,}<extra></extra>",
        ))
        lo5 = base_layout("Session Volume by Content Type", height=320)
        lo5["xaxis"]["title"] = "Sessions"
        lo5["margin"]["r"] = 60
        fig5.update_layout(**lo5)
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        st.markdown(section_label("ACQUISITION CHANNEL — VOLUME vs CHURN"), unsafe_allow_html=True)
        ch_stats = (
            subs.groupby("Acquisition Channel")
            .agg(count=("Subscriber Id", "count"),
                 churn_rate=("churn_flag", "mean"))
            .reset_index()
        )
        fig6 = go.Figure()
        fig6.add_trace(go.Bar(
            name="Subscribers",
            x=ch_stats["Acquisition Channel"],
            y=ch_stats["count"],
            marker=dict(
                color=[CHANNEL_COLORS.get(c, ACCENT_TEAL) for c in ch_stats["Acquisition Channel"]],
                opacity=0.85,
            ),
            hovertemplate="<b>%{x}</b><br>Subscribers: %{y}<extra></extra>",
        ))
        fig6.add_trace(go.Scatter(
            name="Churn %",
            x=ch_stats["Acquisition Channel"],
            y=ch_stats["churn_rate"] * 100,
            mode="lines+markers",
            yaxis="y2",
            line=dict(color=F1_RED, width=2.5),
            marker=dict(size=9, color=F1_RED, line=dict(color=F1_WHITE, width=1.5)),
            hovertemplate="Churn: %{y:.1f}%<extra></extra>",
        ))
        lo6 = base_layout("Acquisition Channel: Volume vs Churn Rate", height=320)
        lo6["yaxis"]["title"]     = "Subscribers"
        lo6["yaxis"]["gridcolor"] = "#252525"
        lo6["yaxis2"] = dict(
            title="Churn %",
            overlaying="y",
            side="right",
            tickfont=dict(color=F1_RED),
            gridcolor="rgba(0,0,0,0)",
        )
        lo6["legend"] = dict(bgcolor="rgba(0,0,0,0)", x=0.01, y=0.99)
        fig6.update_layout(**lo6)
        st.plotly_chart(fig6, use_container_width=True)

    # ── Row 4 ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    col7, col8 = st.columns(2)

    with col7:
        st.markdown(section_label("MONTHLY RECURRING REVENUE — BY PLAN"), unsafe_allow_html=True)
        mrr_pivot = (
            mrr.pivot_table(index="Month", columns="Plan",
                            values="Mrr Usd", aggfunc="sum")
            .fillna(0)
        )
        fig7 = go.Figure()
        for plan, color in PLAN_COLORS.items():
            if plan not in mrr_pivot.columns:
                continue
            fig7.add_trace(go.Scatter(
                x=mrr_pivot.index,
                y=mrr_pivot[plan],
                name=plan,
                mode="lines",
                fill="tonexty",
                line=dict(color=color, width=2.5),
                fillcolor=hex_to_rgba(color, 0.09),
                hovertemplate=f"<b>{plan}</b><br>%{{x|%b %Y}}: $%{{y:,.0f}}<extra></extra>",
            ))
        lo7 = base_layout("MRR by Plan — Jan 2023 to Dec 2024", height=340)
        lo7["xaxis"]["title"] = "Month"
        lo7["yaxis"]["title"] = "MRR (USD)"
        fig7.update_layout(**lo7)
        st.plotly_chart(fig7, use_container_width=True)

    with col8:
        st.markdown(section_label("NPS CATEGORY BY PLAN"), unsafe_allow_html=True)
        nps_plan = (
            subs.groupby(["Plan", "Nps Category"]).size()
            .reset_index(name="count")
        )
        fig8 = px.bar(
            nps_plan, x="Plan", y="count", color="Nps Category",
            color_discrete_map=NPS_COLORS,
            barmode="stack",
            category_orders={
                "Plan": ["Pit Lane", "Podium", "Paddock Club"],
                "Nps Category": ["Detractor", "Passive", "Promoter"],
            },
        )
        lo8 = base_layout("NPS Category by Plan Tier", height=340)
        lo8["yaxis"]["title"] = "Subscribers"
        lo8["xaxis"]["title"] = "Plan"
        fig8.update_layout(**lo8)
        st.plotly_chart(fig8, use_container_width=True)

    # ── Insights ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(section_label("KEY DESCRIPTIVE INSIGHTS"), unsafe_allow_html=True)

    best_plan  = plan_stats.loc[plan_stats["retention_pct"].idxmax(), "Plan"]
    worst_plan = plan_stats.loc[plan_stats["retention_pct"].idxmin(), "Plan"]
    best_ch    = ch_stats.loc[ch_stats["churn_rate"].idxmin(), "Acquisition Channel"]
    top_ct     = ct.iloc[0]["content"]

    i1, i2, i3 = st.columns(3)
    with i1:
        rr = plan_stats.loc[plan_stats["Plan"] == best_plan, "retention_pct"].values[0]
        st.markdown(insight_box(
            f"🏆 <b>{best_plan}</b> achieves the highest retention rate ({rr:.1f}%). "
            f"Premium-tier subscribers show stronger platform commitment — "
            f"higher price correlates with deeper engagement intent and lower churn risk."
        ), unsafe_allow_html=True)
    with i2:
        cr = ch_stats.loc[ch_stats["Acquisition Channel"] == best_ch, "churn_rate"].values[0]
        st.markdown(insight_box(
            f"📣 <b>{best_ch}</b> is the lowest-churn acquisition channel "
            f"({cr*100:.1f}% churn). Reallocating CAC budget toward organic and referral "
            f"sources improves LTV:CAC ratio before the subscriber logs a single session."
        ), unsafe_allow_html=True)
    with i3:
        st.markdown(insight_box(
            f"🎬 <b>{top_ct}</b> drives the highest session volume. "
            f"Surfacing this content type prominently in onboarding flows for new "
            f"<b>{worst_plan}</b> subscribers could reduce the early-dropout spike "
            f"visible in cohort months 1–3."
        ), unsafe_allow_html=True)
