"""
tab3_predictive.py  —  Predictive Analytics
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from theme import (
    base_layout, section_label, insight_box, warn_box, hex_to_rgba,
    F1_RED, F1_WHITE, F1_SILVER, F1_GREY, F1_DGREY, F1_GOLD,
    ACCENT_TEAL, ACCENT_GREEN, ACCENT_AMBER,
    PLAN_COLORS, RISK_COLORS, SEGMENT_COLORS,
)
from model_utils import engineer_features, train_churn_model, get_model_metrics, segment_customers


@st.cache_data(show_spinner=False)
def _run_pipeline(subs: pd.DataFrame, sess: pd.DataFrame) -> tuple:
    df      = engineer_features(subs, sess)
    out     = train_churn_model(df)
    clf, X_tr, X_te, y_tr, y_te, y_pred, y_prob, imp_df, df_sc = out
    metrics = get_model_metrics(y_te, y_pred, y_prob)
    df_sc   = segment_customers(df_sc)
    return df_sc, metrics, imp_df, y_te, y_pred, y_prob


def render(subs: pd.DataFrame, sess: pd.DataFrame, mrr: pd.DataFrame) -> None:
    st.markdown("## Predictive — *Who Will Churn Next?*")
    st.markdown(
        "Random Forest churn classifier · feature importance · "
        "risk scoring · KMeans behavioural segmentation."
    )
    st.markdown("---")

    with st.spinner("🏎  Training churn model on real subscriber data…"):
        df, metrics, imp_df, y_te, y_pred, y_prob = _run_pipeline(subs, sess)

    # ── Model scorecard ───────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  f"{metrics['accuracy']:.1%}")
    m2.metric("Precision", f"{metrics['precision']:.1%}")
    m3.metric("Recall",    f"{metrics['recall']:.1%}")
    m4.metric("F1 Score",  f"{metrics['f1']:.1%}")
    m5.metric("ROC-AUC",   f"{metrics['auc']:.3f}", "Target ≥ 0.75")
    st.markdown("---")

    # ── Row 1: Feature importance  |  ROC curve ───────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(section_label("FEATURE IMPORTANCE — WHAT DRIVES CHURN?"), unsafe_allow_html=True)
        med = float(imp_df["importance"].median())
        fig1 = go.Figure(go.Bar(
            y=imp_df["feature"],
            x=imp_df["importance"],
            orientation="h",
            marker=dict(
                color=[F1_RED if v > med else F1_SILVER for v in imp_df["importance"]],
                line=dict(color=F1_DGREY, width=0.5),
            ),
            text=[f"{v:.3f}" for v in imp_df["importance"]],
            textposition="outside",
            textfont=dict(color=F1_WHITE, size=10),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ))
        lo1 = base_layout("Random Forest — Feature Importance", height=460)
        lo1["xaxis"]["title"] = "Gini Importance"
        lo1["margin"]["r"] = 70
        fig1.update_layout(**lo1)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown(section_label("ROC CURVE — CHURN CLASSIFIER"), unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=metrics["fpr"],
            y=metrics["tpr"],
            mode="lines",
            name=f"RF Model  (AUC = {metrics['auc']:.3f})",
            line=dict(color=F1_RED, width=3),
            fill="tozeroy",
            fillcolor="rgba(232,0,45,0.10)",
        ))
        fig2.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Baseline",
            line=dict(color=F1_SILVER, width=1.5, dash="dash"),
        ))
        lo2 = base_layout(f"ROC Curve  ·  AUC = {metrics['auc']:.3f}", height=460)
        lo2["xaxis"]["title"] = "False Positive Rate"
        lo2["yaxis"]["title"] = "True Positive Rate"
        fig2.update_layout(**lo2)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Confusion matrix  |  Risk distribution ─────────────────────────
    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown(section_label("CONFUSION MATRIX — TEST SET"), unsafe_allow_html=True)
        cm = metrics["cm"]
        fig3 = go.Figure(go.Heatmap(
            z=cm,
            x=["Pred: Active", "Pred: Churned"],
            y=["Actual: Active", "Actual: Churned"],
            colorscale=[[0,"#2A2A2A"],[1,"#FDA4AF"]],
            text=cm,
            texttemplate="<b>%{text}</b>",
            textfont=dict(size=26, color=F1_WHITE),
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            showscale=False,
        ))
        lo3 = base_layout("Confusion Matrix", height=320)
        lo3["xaxis"]["title"] = "Predicted"
        lo3["yaxis"]["title"] = "Actual"
        fig3.update_layout(**lo3)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown(
            section_label("CHURN RISK DISTRIBUTION — ACTIVE SUBSCRIBERS"),
            unsafe_allow_html=True,
        )
        active = df[df["churn_flag"] == 0].copy()
        active["risk_label"] = pd.cut(
            active["churn_prob"],
            bins=[0, 0.33, 0.66, 1.0],
            labels=["Low Risk", "Medium Risk", "High Risk"],
        )
        rc = active["risk_label"].value_counts()
        fig4 = go.Figure(go.Pie(
            labels=rc.index.tolist(),
            values=rc.values.tolist(),
            hole=0.52,
            marker=dict(
                colors=[RISK_COLORS.get(str(l), F1_SILVER) for l in rc.index],
                line=dict(color=F1_DGREY, width=2),
            ),
            textinfo="label+percent+value",
            textfont=dict(color=F1_WHITE, size=12),
            hovertemplate="<b>%{label}</b><br>%{value} subs (%{percent})<extra></extra>",
        ))
        fig4.update_layout(**base_layout("Active Subscribers — Churn Risk Tier", height=320))
        fig4.add_annotation(
            text=f"<b>{len(active):,}</b><br>active",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color=F1_WHITE, size=14),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Row 3: Churn prob by plan  |  by NPS ──────────────────────────────────
    st.markdown("---")
    col5, col6 = st.columns(2)

    with col5:
        st.markdown(
            section_label("CHURN PROBABILITY DISTRIBUTION — BY PLAN"),
            unsafe_allow_html=True,
        )
        fig5 = go.Figure()
        for plan, color in PLAN_COLORS.items():
            vals = active[active["Plan"] == plan]["churn_prob"].dropna()
            fig5.add_trace(go.Violin(
                y=vals,
                name=plan,
                fillcolor=hex_to_rgba(color, 0.25),
                line_color=color,
                meanline_visible=True,
                box_visible=True,
                hoverinfo="y+name",
            ))
        lo5 = base_layout("Predicted Churn Probability by Plan", height=360)
        lo5["yaxis"]["title"]      = "Predicted Churn Probability"
        lo5["yaxis"]["tickformat"] = ".0%"
        fig5.update_layout(**lo5)
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        st.markdown(section_label("CHURN PROBABILITY vs NPS SCORE"), unsafe_allow_html=True)
        nps_churn = (
            active.groupby("Nps Score")["churn_prob"]
            .mean()
            .reset_index()
        )
        fig6 = go.Figure(go.Scatter(
            x=nps_churn["Nps Score"],
            y=nps_churn["churn_prob"],
            mode="lines+markers",
            line=dict(color=F1_RED, width=2.5),
            marker=dict(color=F1_RED, size=9, line=dict(color=F1_WHITE, width=1.5)),
            fill="tozeroy",
            fillcolor="rgba(232,0,45,0.08)",
            hovertemplate="NPS %{x}<br>Avg Churn Prob: %{y:.1%}<extra></extra>",
        ))
        lo6 = base_layout("Avg Predicted Churn Probability by NPS Score", height=360)
        lo6["xaxis"]["title"]      = "NPS Score (0–10)"
        lo6["yaxis"]["title"]      = "Avg Predicted Churn Probability"
        lo6["yaxis"]["tickformat"] = ".0%"
        fig6.update_layout(**lo6)
        st.plotly_chart(fig6, use_container_width=True)

    # ── High-risk watchlist ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        section_label("HIGH-RISK WATCHLIST — TOP 30 ACTIVE SUBSCRIBERS"),
        unsafe_allow_html=True,
    )
    st.markdown(
        "Active subscribers with predicted churn probability ≥ 45%, "
        "ranked by Priority Score = churn probability × monthly price."
    )
    watchlist = active[active["churn_prob"] >= 0.45].copy()
    watchlist["priority_score"] = (watchlist["churn_prob"] * watchlist["Monthly Price Usd"]).round(2)
    watchlist = watchlist.sort_values("priority_score", ascending=False).head(30)

    disp = watchlist[[
        "Subscriber Id", "Plan", "Region", "Monthly Price Usd",
        "avg_engagement", "avg_duration", "churn_prob", "priority_score",
    ]].copy()
    disp.columns = [
        "Subscriber", "Plan", "Region", "Price ($)",
        "Avg Engagement", "Avg Duration (min)", "Churn Prob", "Priority Score",
    ]
    disp["Churn Prob"]         = disp["Churn Prob"].map("{:.1%}".format)
    disp["Avg Engagement"]     = disp["Avg Engagement"].map("{:.1f}".format)
    disp["Avg Duration (min)"] = disp["Avg Duration (min)"].map("{:.1f}".format)
    disp["Priority Score"]     = disp["Priority Score"].map("{:.2f}".format)
    st.dataframe(disp, use_container_width=True, height=380)

    # ── KMeans segmentation ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(section_label("KMEANS CUSTOMER SEGMENTATION — 4 CLUSTERS"), unsafe_allow_html=True)
    st.markdown(
        "##### Segmented on: avg engagement, avg duration, total sessions, "
        "tenure months, mobile %, high-engagement session %"
    )
    col7, col8 = st.columns([1, 2])

    seg_sum = (
        df.groupby("segment_label")
        .agg(count=("Subscriber Id", "count"),
             avg_eng=("avg_engagement", "mean"),
             avg_dur=("avg_duration", "mean"),
             churn_rate=("churn_flag", "mean"))
        .reset_index()
        .sort_values("avg_eng", ascending=False)
    )

    with col7:
        fig7 = go.Figure(go.Bar(
            x=seg_sum["count"],
            y=seg_sum["segment_label"],
            orientation="h",
            marker=dict(
                color=[SEGMENT_COLORS.get(s, F1_SILVER) for s in seg_sum["segment_label"]],
                line=dict(color=F1_DGREY, width=1),
            ),
            text=seg_sum["count"].tolist(),
            textposition="outside",
            textfont=dict(color=F1_WHITE),
            customdata=seg_sum[["avg_eng", "churn_rate"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>Subscribers: %{x}<br>"
                "Avg Engagement: %{customdata[0]:.1f}<br>"
                "Churn Rate: %{customdata[1]:.1%}<extra></extra>"
            ),
        ))
        lo7 = base_layout("Segment Sizes", height=300)
        lo7["xaxis"]["title"] = "Subscribers"
        lo7["margin"]["r"] = 60
        fig7.update_layout(**lo7)
        st.plotly_chart(fig7, use_container_width=True)

    with col8:
        samp = df.sample(min(700, len(df)), random_state=42)
        fig8 = px.scatter(
            samp,
            x="avg_duration",
            y="avg_engagement",
            color="segment_label",
            color_discrete_map=SEGMENT_COLORS,
            size="Tenure Months",
            size_max=18,
            opacity=0.72,
            hover_data={
                "Subscriber Id": True,
                "Plan": True,
                "churn_prob": ":.1%",
                "Tenure Months": True,
            },
            labels={
                "avg_duration":   "Avg Session Duration (min)",
                "avg_engagement": "Avg Engagement Score",
                "segment_label":  "Segment",
            },
        )
        fig8.update_layout(**base_layout(
            "Segment Scatter — Engagement vs Duration  (bubble size = Tenure)", height=360))
        st.plotly_chart(fig8, use_container_width=True)

    # ── Segment comparisons ────────────────────────────────────────────────────
    st.markdown("---")
    col9, col10 = st.columns(2)

    with col9:
        st.markdown(section_label("CHURN RATE BY SEGMENT"), unsafe_allow_html=True)
        fig9 = go.Figure(go.Bar(
            x=seg_sum["segment_label"],
            y=seg_sum["churn_rate"] * 100,
            marker=dict(
                color=[SEGMENT_COLORS.get(s, F1_SILVER) for s in seg_sum["segment_label"]],
                line=dict(color=F1_DGREY, width=1),
            ),
            text=[f"{v*100:.1f}%" for v in seg_sum["churn_rate"]],
            textposition="outside",
            textfont=dict(color=F1_WHITE, size=12),
            hovertemplate="<b>%{x}</b><br>Churn Rate: %{y:.1f}%<extra></extra>",
        ))
        lo9 = base_layout("Churn Rate by KMeans Segment", height=320)
        lo9["yaxis"]["title"] = "Churn Rate (%)"
        lo9["yaxis"]["range"] = [0, float(seg_sum["churn_rate"].max()) * 140]
        fig9.update_layout(**lo9)
        st.plotly_chart(fig9, use_container_width=True)

    with col10:
        st.markdown(section_label("AVG ENGAGEMENT SCORE BY SEGMENT"), unsafe_allow_html=True)
        fig10 = go.Figure(go.Bar(
            x=seg_sum["segment_label"],
            y=seg_sum["avg_eng"],
            marker=dict(
                color=[SEGMENT_COLORS.get(s, F1_SILVER) for s in seg_sum["segment_label"]],
                line=dict(color=F1_DGREY, width=1),
            ),
            text=[f"{v:.1f}" for v in seg_sum["avg_eng"]],
            textposition="outside",
            textfont=dict(color=F1_WHITE, size=12),
            hovertemplate="<b>%{x}</b><br>Avg Engagement: %{y:.1f}<extra></extra>",
        ))
        lo10 = base_layout("Avg Engagement Score by Segment", height=320)
        lo10["yaxis"]["title"] = "Avg Engagement Score"
        lo10["yaxis"]["range"] = [0, float(seg_sum["avg_eng"].max()) * 1.22]
        fig10.update_layout(**lo10)
        st.plotly_chart(fig10, use_container_width=True)

    # ── Insights ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(section_label("KEY PREDICTIVE INSIGHTS"), unsafe_allow_html=True)

    top_feat    = imp_df.iloc[-1]["feature"]
    high_risk_n = len(active[active["churn_prob"] >= 0.66])
    dormant_n   = len(df[df["segment_label"] == "Dormant"])

    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown(insight_box(
            f"🎯 <b>'{top_feat}'</b> is the single strongest churn predictor. "
            f"Interventions targeting this variable will generate the highest incremental "
            f"improvement in retention across all plan tiers."
        ), unsafe_allow_html=True)
    with i2:
        st.markdown(insight_box(
            f"🚨 <b>{high_risk_n} active subscribers</b> have predicted churn probability ≥ 66%. "
            f"These are the highest-priority intervention targets — see the watchlist above "
            f"for Priority Score ranking to decide where to spend retention budget first."
        ), unsafe_allow_html=True)
    with i3:
        st.markdown(warn_box(
            f"😴 <b>{dormant_n} subscribers</b> are in the Dormant segment — "
            f"low engagement, short sessions, high churn probability. "
            f"This group needs a re-engagement campaign before probability crosses 70%."
        ), unsafe_allow_html=True)
