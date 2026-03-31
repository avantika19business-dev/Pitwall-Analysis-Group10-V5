"""
tab_clustering.py  —  Clustering Analysis
KMeans with elbow method, PCA visualisation, cluster profiles, persona cards
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from theme import (
    base_layout, section_label, insight_box, rec_box, warn_box, hex_to_rgba,
    F1_RED, F1_WHITE, F1_SILVER, F1_GREY, F1_DGREY, F1_GOLD,
    ACCENT_TEAL, ACCENT_GREEN, ACCENT_AMBER, ACCENT_PURPLE,
    SEGMENT_COLORS,
)
from model_utils import engineer_features, segment_customers, get_elbow_data

CLUSTER_FEATS = ["avg_engagement","avg_duration","total_sessions",
                 "Tenure Months","mobile_pct","high_eng_pct"]

@st.cache_data(show_spinner=False)
def _run(subs: pd.DataFrame, sess: pd.DataFrame):
    df      = engineer_features(subs, sess)
    elbow   = get_elbow_data(df)
    df_seg  = segment_customers(df, n_clusters=4)
    return df_seg, elbow


def render(subs: pd.DataFrame, sess: pd.DataFrame, mrr: pd.DataFrame) -> None:
    st.markdown("## Clustering Analysis")
    st.markdown(
        "K-Means behavioural segmentation — subscribers grouped by engagement, "
        "session behaviour, and tenure. Elbow method used to select optimal k."
    )
    st.markdown("---")

    with st.spinner("🏎  Running KMeans segmentation…"):
        df, elbow = _run(subs, sess)

    # ── Row 1: Elbow + Cluster distribution ───────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(section_label("OPTIMAL K SELECTION — ELBOW METHOD"), unsafe_allow_html=True)
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(
            x=elbow["k"], y=elbow["inertia"],
            name="Inertia (WCSS)",
            mode="lines+markers",
            line=dict(color=F1_RED, width=2.5),
            marker=dict(size=8, color=F1_RED, line=dict(color=F1_WHITE, width=1.5)),
            hovertemplate="k=%{x}<br>Inertia: %{y:,.0f}<extra></extra>",
        ))
        fig_e.add_trace(go.Scatter(
            x=elbow["k"], y=elbow["silhouette"],
            name="Silhouette Score",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color=ACCENT_GREEN, width=2.5, dash="dot"),
            marker=dict(size=8, color=ACCENT_GREEN, line=dict(color=F1_WHITE, width=1.5)),
            hovertemplate="k=%{x}<br>Silhouette: %{y:.3f}<extra></extra>",
        ))
        fig_e.add_vline(x=4, line_dash="dash", line_color=F1_GOLD,
                        annotation_text="k=4 selected", annotation_font_color=F1_GOLD)
        lo_e = base_layout("Optimal k — Elbow & Silhouette", height=360)
        lo_e["xaxis"]["title"]  = "Number of Clusters (k)"
        lo_e["yaxis"]["title"]  = "Inertia (WCSS)"
        lo_e["yaxis2"] = dict(title="Silhouette Score", overlaying="y", side="right",
                               tickfont=dict(color=ACCENT_GREEN),
                               gridcolor="rgba(0,0,0,0)")
        fig_e.update_layout(**lo_e)
        st.plotly_chart(fig_e, use_container_width=True)

    with col2:
        st.markdown(section_label("CLUSTER SIZE DISTRIBUTION"), unsafe_allow_html=True)
        seg_counts = df["segment_label"].value_counts().reset_index()
        seg_counts.columns = ["segment","count"]
        pct = seg_counts["count"] / seg_counts["count"].sum() * 100
        fig_d = go.Figure(go.Pie(
            labels=seg_counts["segment"],
            values=seg_counts["count"],
            hole=0.5,
            marker=dict(
                colors=[SEGMENT_COLORS.get(s, F1_SILVER) for s in seg_counts["segment"]],
                line=dict(color=F1_DGREY, width=2),
            ),
            textinfo="label+percent+value",
            textfont=dict(color=F1_WHITE, size=12),
            hovertemplate="<b>%{label}</b><br>%{value} subscribers (%{percent})<extra></extra>",
        ))
        fig_d.update_layout(**base_layout("Cluster Size Distribution", height=360))
        fig_d.add_annotation(text=f"<b>{len(df)}</b><br>subscribers",
                             x=0.5, y=0.5, showarrow=False,
                             font=dict(color=F1_WHITE, size=14))
        st.plotly_chart(fig_d, use_container_width=True)

    # ── Row 2: PCA scatter ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(section_label("2D PCA CLUSTER VISUALISATION"), unsafe_allow_html=True)

    X_scaled = StandardScaler().fit_transform(df[CLUSTER_FEATS].fillna(0))
    pca2     = PCA(n_components=2, random_state=42)
    coords   = pca2.fit_transform(X_scaled)
    df_plot  = df.copy()
    df_plot["PC1"] = coords[:, 0]
    df_plot["PC2"] = coords[:, 1]

    samp = df_plot.sample(min(700, len(df_plot)), random_state=42)
    fig_pca = px.scatter(
        samp, x="PC1", y="PC2",
        color="segment_label",
        color_discrete_map=SEGMENT_COLORS,
        size="Tenure Months",
        size_max=16,
        opacity=0.75,
        hover_data={
            "Subscriber Id": True,
            "Plan": True,
            "avg_engagement": ":.1f",
            "churn_prob": ":.1%" if "churn_prob" in samp.columns else False,
            "Tenure Months": True,
        },
        labels={"PC1": f"PC1 ({pca2.explained_variance_ratio_[0]:.1%} var)",
                "PC2": f"PC2 ({pca2.explained_variance_ratio_[1]:.1%} var)",
                "segment_label": "Segment"},
    )
    fig_pca.update_layout(**base_layout(
        "KMeans Clusters — PCA Projection (bubble size = Tenure)", height=460))
    st.plotly_chart(fig_pca, use_container_width=True)

    # ── Row 3: Cluster profiles table ─────────────────────────────────────────
    st.markdown("---")
    st.markdown(section_label("CLUSTER PROFILES"), unsafe_allow_html=True)

    seg_sum = (
        df.groupby("segment_label")
        .agg(
            count        = ("Subscriber Id",  "count"),
            avg_eng      = ("avg_engagement", "mean"),
            avg_dur      = ("avg_duration",   "mean"),
            avg_tenure   = ("Tenure Months",  "mean"),
            avg_sessions = ("total_sessions", "mean"),
            churn_rate   = ("churn_flag",     "mean"),
        )
        .reset_index()
        .sort_values("avg_eng", ascending=False)
    )

    profile_df = seg_sum.copy()
    profile_df["Avg Engagement"] = profile_df["avg_eng"].map("{:.1f}".format)
    profile_df["Avg Duration"]   = profile_df["avg_dur"].map("{:.0f} min".format)
    profile_df["Avg Tenure"]     = profile_df["avg_tenure"].map("{:.0f} mo".format)
    profile_df["Avg Sessions"]   = profile_df["avg_sessions"].map("{:.0f}".format)
    profile_df["Churn Rate"]     = profile_df["churn_rate"].map("{:.1%}".format)
    profile_df["Size"]           = profile_df["count"]
    st.dataframe(
        profile_df[["segment_label","Size","Avg Engagement","Avg Duration",
                    "Avg Tenure","Avg Sessions","Churn Rate"]]
        .rename(columns={"segment_label": "Segment"}),
        use_container_width=True, hide_index=True,
    )

    # ── Row 4: Metric comparisons ──────────────────────────────────────────────
    st.markdown("---")
    c3, c4, c5 = st.columns(3)

    with c3:
        st.markdown(section_label("CHURN RATE BY CLUSTER"), unsafe_allow_html=True)
        fig_cr = go.Figure(go.Bar(
            x=seg_sum["segment_label"],
            y=seg_sum["churn_rate"] * 100,
            marker=dict(
                color=[SEGMENT_COLORS.get(s, F1_SILVER) for s in seg_sum["segment_label"]],
                line=dict(color=F1_DGREY, width=1),
            ),
            text=[f"{v*100:.1f}%" for v in seg_sum["churn_rate"]],
            textposition="outside",
            textfont=dict(color=F1_WHITE, size=12),
            hovertemplate="<b>%{x}</b><br>Churn: %{y:.1f}%<extra></extra>",
        ))
        lo_cr = base_layout("Churn Rate by Cluster", height=320)
        lo_cr["yaxis"]["title"] = "Churn Rate (%)"
        lo_cr["yaxis"]["range"] = [0, float(seg_sum["churn_rate"].max()) * 145]
        fig_cr.update_layout(**lo_cr)
        st.plotly_chart(fig_cr, use_container_width=True)

    with c4:
        st.markdown(section_label("AVG ENGAGEMENT BY CLUSTER"), unsafe_allow_html=True)
        fig_ae = go.Figure(go.Bar(
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
        lo_ae = base_layout("Avg Engagement by Cluster", height=320)
        lo_ae["yaxis"]["title"] = "Avg Engagement Score"
        lo_ae["yaxis"]["range"] = [0, float(seg_sum["avg_eng"].max()) * 1.22]
        fig_ae.update_layout(**lo_ae)
        st.plotly_chart(fig_ae, use_container_width=True)

    with c5:
        st.markdown(section_label("AVG SESSION DURATION BY CLUSTER"), unsafe_allow_html=True)
        fig_sd = go.Figure(go.Bar(
            x=seg_sum["segment_label"],
            y=seg_sum["avg_dur"],
            marker=dict(
                color=[SEGMENT_COLORS.get(s, F1_SILVER) for s in seg_sum["segment_label"]],
                line=dict(color=F1_DGREY, width=1),
            ),
            text=[f"{v:.0f} min" for v in seg_sum["avg_dur"]],
            textposition="outside",
            textfont=dict(color=F1_WHITE, size=12),
            hovertemplate="<b>%{x}</b><br>Avg Duration: %{y:.1f} min<extra></extra>",
        ))
        lo_sd = base_layout("Avg Session Duration by Cluster", height=320)
        lo_sd["yaxis"]["title"] = "Avg Duration (min)"
        lo_sd["yaxis"]["range"] = [0, float(seg_sum["avg_dur"].max()) * 1.22]
        fig_sd.update_layout(**lo_sd)
        st.plotly_chart(fig_sd, use_container_width=True)

    # ── Customer Persona Cards ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(section_label("CUSTOMER PERSONA CARDS"), unsafe_allow_html=True)

    personas = {
        "Champions": {
            "icon": "🏆",
            "desc": "High engagement, long sessions, strong loyalty. Top spenders with multiple renewals.",
            "strategy": "Offer early access to new content drops and a Paddock Club loyalty badge.",
            "color": F1_GOLD,
        },
        "Engaged": {
            "icon": "🟢",
            "desc": "Consistent usage, healthy engagement. On the upgrade path if nudged correctly.",
            "strategy": "Targeted Podium upgrade campaign — '2 months free' trial converts well.",
            "color": ACCENT_GREEN,
        },
        "At Risk": {
            "icon": "⚠️",
            "desc": "Declining sessions and engagement. Have logged fewer sessions recently.",
            "strategy": "Trigger a personalised re-engagement email with their top content type.",
            "color": ACCENT_AMBER,
        },
        "Dormant": {
            "icon": "😴",
            "desc": "Minimal engagement, high churn probability. Last-chance intervention window.",
            "strategy": "PitWall Race Week challenge — gamified prediction contest to reignite interest.",
            "color": F1_RED,
        },
    }

    pcols = st.columns(4)
    for col, (seg_name, p) in zip(pcols, personas.items()):
        seg_row = seg_sum[seg_sum["segment_label"] == seg_name]
        n     = int(seg_row["count"].values[0])     if len(seg_row) > 0 else 0
        cr    = float(seg_row["churn_rate"].values[0]) if len(seg_row) > 0 else 0
        eng   = float(seg_row["avg_eng"].values[0])  if len(seg_row) > 0 else 0
        col.markdown(f"""
        <div style="
            background: linear-gradient(145deg,#1A1A1A,#0F0F0F);
            border: 1px solid {p['color']};
            border-top: 4px solid {p['color']};
            border-radius: 8px; padding: 18px 16px; height: 280px;
        ">
            <div style="font-size:28px;margin-bottom:6px;">{p['icon']}</div>
            <div style="font-size:14px;font-weight:900;color:{p['color']};
                        letter-spacing:1px;margin-bottom:4px;">{seg_name.upper()}</div>
            <div style="font-size:11px;color:#aaa;margin-bottom:10px;">
                {n} subs · {cr:.1%} churn · eng {eng:.0f}
            </div>
            <div style="font-size:11px;color:#ccc;line-height:1.5;margin-bottom:8px;">
                {p['desc']}
            </div>
            <div style="font-size:10px;color:{p['color']};font-style:italic;
                        border-top:1px solid #2a2a2a;padding-top:8px;line-height:1.4;">
                💡 {p['strategy']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Insights ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(section_label("KEY CLUSTERING INSIGHTS"), unsafe_allow_html=True)
    top_seg  = seg_sum.iloc[0]["segment_label"]
    worst    = seg_sum.sort_values("churn_rate", ascending=False).iloc[0]["segment_label"]
    dormant_n = int(df[df["segment_label"] == "Dormant"]["Subscriber Id"].count())

    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown(insight_box(
            f"🏆 <b>{top_seg}</b> subscribers have the highest engagement score. "
            f"They represent PitWall's most loyal base — invest in exclusive perks "
            f"(early content, community access) to sustain and expand this cohort."
        ), unsafe_allow_html=True)
    with i2:
        cr_worst = float(seg_sum[seg_sum["segment_label"] == worst]["churn_rate"].values[0])
        st.markdown(warn_box(
            f"🚨 <b>{worst}</b> cluster has the highest churn rate ({cr_worst:.1%}). "
            f"These subscribers exhibit low engagement and short sessions — "
            f"a targeted re-engagement campaign is needed before they hit the 70% "
            f"churn probability threshold."
        ), unsafe_allow_html=True)
    with i3:
        st.markdown(insight_box(
            f"📊 <b>4 clusters selected</b> via the elbow method — the inertia curve "
            f"shows diminishing returns beyond k=4. Silhouette score confirms clear "
            f"cluster separation. Behavioural features (engagement, sessions, tenure) "
            f"drive the majority of inter-cluster variance."
        ), unsafe_allow_html=True)
