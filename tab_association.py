"""
tab_association.py  —  Association Rule Mining
Content-type co-occurrence basket analysis with Apriori-style rules
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from theme import (
    base_layout, section_label, insight_box, rec_box, hex_to_rgba,
    F1_RED, F1_WHITE, F1_SILVER, F1_GREY, F1_DGREY, F1_GOLD,
    ACCENT_TEAL, ACCENT_GREEN, ACCENT_AMBER,
)
from model_utils import mine_association_rules


@st.cache_data(show_spinner=False)
def _run(sess: pd.DataFrame, subs: pd.DataFrame,
         min_sup: float, min_conf: float, min_lift: float):
    return mine_association_rules(sess, subs, min_sup, min_conf, min_lift)


def render(subs: pd.DataFrame, sess: pd.DataFrame, mrr: pd.DataFrame) -> None:
    st.markdown("## Association Rule Mining")
    st.markdown(
        "Which content types, plan tiers, and churn statuses are consumed together? "
        "Apriori algorithm applied to subscriber-level content baskets."
    )
    st.markdown("---")

    # ── Controls ─────────────────────────────────────────────────────────────
    st.markdown(section_label("APRIORI CONTROLS"), unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    min_sup  = c1.slider("Min Support",    0.02, 0.30, 0.05, 0.01)
    min_conf = c2.slider("Min Confidence", 0.10, 0.90, 0.30, 0.05)
    min_lift = c3.slider("Min Lift",       1.0,  3.0,  1.0,  0.1)

    with st.spinner("⛏️  Mining association rules…"):
        rules_df, freq_df = _run(sess, subs, min_sup, min_conf, min_lift)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    n_rules   = len(rules_df)
    avg_conf  = float(rules_df["confidence"].mean()) if n_rules > 0 else 0
    avg_lift  = float(rules_df["lift"].mean())       if n_rules > 0 else 0
    max_lift  = float(rules_df["lift"].max())        if n_rules > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Rules Found",    f"{n_rules:,}")
    k2.metric("Avg Confidence",       f"{avg_conf:.1%}")
    k3.metric("Avg Lift",             f"{avg_lift:.2f}")
    k4.metric("Max Lift",             f"{max_lift:.2f}", "Higher = stronger association")
    st.markdown("---")

    if n_rules == 0:
        st.warning("No rules found with current thresholds. Try lowering Min Support or Min Confidence.")
        return

    # ── Rules table ───────────────────────────────────────────────────────────
    st.markdown(section_label("ASSOCIATION RULES TABLE"), unsafe_allow_html=True)
    disp = rules_df.head(30).copy()
    disp["support"]    = disp["support"].map("{:.3f}".format)
    disp["confidence"] = disp["confidence"].map("{:.3f}".format)
    disp["lift"]       = disp["lift"].map("{:.3f}".format)
    disp["conviction"] = disp["conviction"].map("{:.3f}".format)
    disp.columns = [c.title() for c in disp.columns]
    st.dataframe(disp, use_container_width=True, hide_index=True, height=360)

    st.markdown("---")
    c1, c2 = st.columns(2)

    # ── Support vs Confidence scatter ─────────────────────────────────────────
    with c1:
        st.markdown(section_label("SUPPORT vs CONFIDENCE  (size = Lift)"), unsafe_allow_html=True)
        fig_sc = px.scatter(
            rules_df,
            x="support", y="confidence",
            size="lift", color="lift",
            color_continuous_scale=[[0,"#C4B5FD"],[0.5,"#BAE6FD"],[1,"#86EFAC"]],
            hover_data=["antecedent","consequent","lift","conviction"],
            labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"},
            size_max=22,
        )
        fig_sc.update_coloraxes(
            colorbar=dict(title="Lift", tickfont=dict(color=F1_SILVER))
        )
        fig_sc.update_layout(**base_layout("Support vs Confidence  (bubble = Lift)", height=400))
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── Top rules by lift ──────────────────────────────────────────────────────
    with c2:
        st.markdown(section_label("TOP 15 RULES BY LIFT"), unsafe_allow_html=True)
        top15 = rules_df.head(15).copy()
        top15["rule"] = top15["antecedent"] + " → " + top15["consequent"]
        top15 = top15.sort_values("lift", ascending=True)
        fig_lift = go.Figure(go.Bar(
            y=top15["rule"],
            x=top15["lift"],
            orientation="h",
            marker=dict(
                color=top15["lift"].tolist(),
                colorscale=[[0,"#BAE6FD"],[0.5,"#C4B5FD"],[1,"#FDA4AF"]],
                showscale=True,
                colorbar=dict(title="Lift", tickfont=dict(color=F1_SILVER), len=0.7),
            ),
            customdata=top15[["confidence","support"]].values,
            text=[f"{v:.3f}" for v in top15["lift"]],
            textposition="outside",
            textfont=dict(color=F1_WHITE, size=10),
            hovertemplate=(
                "<b>%{y}</b><br>Lift: %{x:.3f}<br>"
                "Confidence: %{customdata[0]:.3f}<br>"
                "Support: %{customdata[1]:.3f}<extra></extra>"
            ),
        ))
        lo_lift = base_layout("Top 15 Rules by Lift", height=400)
        lo_lift["xaxis"]["title"] = "Lift"
        lo_lift["margin"]["r"]    = 70
        fig_lift.update_layout(**lo_lift)
        st.plotly_chart(fig_lift, use_container_width=True)

    # ── Item frequency ─────────────────────────────────────────────────────────
    st.markdown("---")
    c3, c4 = st.columns(2)

    with c3:
        st.markdown(section_label("ITEM FREQUENCY — MOST COMMON BASKET ITEMS"), unsafe_allow_html=True)
        top_items = freq_df.head(15).sort_values("support", ascending=True)
        fig_freq = go.Figure(go.Bar(
            y=top_items["item"],
            x=top_items["support"],
            orientation="h",
            marker=dict(
                color=top_items["support"].tolist(),
                colorscale=[[0,"#1C1C1C"],[0.5,"#BAE6FD"],[1,"#C4B5FD"]],
            ),
            text=[f"{v:.3f}" for v in top_items["support"]],
            textposition="outside",
            textfont=dict(color=F1_WHITE, size=10),
            hovertemplate="<b>%{y}</b><br>Support: %{x:.3f}<extra></extra>",
        ))
        lo_freq = base_layout("Item Frequency (Support)", height=400)
        lo_freq["xaxis"]["title"] = "Support"
        lo_freq["margin"]["r"]    = 80
        fig_freq.update_layout(**lo_freq)
        st.plotly_chart(fig_freq, use_container_width=True)

    # ── Co-occurrence heatmap ──────────────────────────────────────────────────
    with c4:
        st.markdown(section_label("CONTENT CO-OCCURRENCE HEATMAP"), unsafe_allow_html=True)
        content_types = sess["Content Type"].unique().tolist()
        # Build co-occurrence matrix from rules
        co_mat = pd.DataFrame(0.0, index=content_types, columns=content_types)
        for _, row in rules_df.iterrows():
            ant, con = row["antecedent"], row["consequent"]
            if ant in content_types and con in content_types:
                co_mat.loc[ant, con] = row["confidence"]
                co_mat.loc[con, ant] = row["confidence"]

        fig_heat = go.Figure(go.Heatmap(
            z=co_mat.values,
            x=co_mat.columns.tolist(),
            y=co_mat.index.tolist(),
            colorscale=[[0,"#1C1C1C"],[0.5,"#BAE6FD"],[1,"#86EFAC"]],
            text=np.round(co_mat.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=11, color=F1_WHITE),
            hovertemplate="<b>%{y} → %{x}</b><br>Confidence: %{z:.3f}<extra></extra>",
            colorbar=dict(title="Confidence", tickfont=dict(color=F1_SILVER)),
        ))
        fig_heat.update_layout(**base_layout("Content Type Co-occurrence (Confidence)", height=400))
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Top actionable rule cards ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown(section_label("TOP ACTIONABLE RULES"), unsafe_allow_html=True)

    # Filter for content-only rules (exclude Plan/Churn items for clarity)
    content_rules = rules_df[
        ~rules_df["antecedent"].str.startswith("Plan:") &
        ~rules_df["antecedent"].str.startswith("Churn:") &
        ~rules_df["consequent"].str.startswith("Plan:") &
        ~rules_df["consequent"].str.startswith("Churn:")
    ].head(4)

    if len(content_rules) == 0:
        content_rules = rules_df.head(4)

    rcols = st.columns(min(4, len(content_rules)))
    colors_cycle = [F1_RED, ACCENT_TEAL, ACCENT_GREEN, ACCENT_AMBER]
    for i, (_, row) in enumerate(content_rules.iterrows()):
        color = colors_cycle[i % len(colors_cycle)]
        rcols[i].markdown(f"""
        <div style="
            background: linear-gradient(145deg,#1A1A1A,#0F0F0F);
            border: 1px solid {color}; border-top: 4px solid {color};
            border-radius: 8px; padding: 16px 14px;
        ">
            <div style="font-size:11px;font-weight:700;color:{color};
                        letter-spacing:1px;margin-bottom:8px;text-transform:uppercase;">
                Rule #{i+1}
            </div>
            <div style="font-size:12px;color:#F5F5F5;font-weight:700;margin-bottom:6px;">
                {row['antecedent']}
                <span style="color:{color}"> → </span>
                {row['consequent']}
            </div>
            <div style="font-size:10px;color:#aaa;line-height:1.8;">
                Confidence: <b style="color:{color}">{float(row['confidence']):.1%}</b><br>
                Lift: <b style="color:{color}">{float(row['lift']):.2f}x</b><br>
                Support: <b>{float(row['support']):.3f}</b>
            </div>
            <div style="font-size:10px;color:#888;margin-top:8px;font-style:italic;
                        border-top:1px solid #2a2a2a;padding-top:6px;">
                Subscribers who consume {row['antecedent']} are 
                {float(row['lift']):.2f}× more likely to also consume {row['consequent']}.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Insights ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(section_label("KEY ASSOCIATION INSIGHTS"), unsafe_allow_html=True)

    # Churn-linked rules
    churn_rules = rules_df[rules_df["consequent"] == "Churn:Yes"].head(1)
    churn_ant   = churn_rules.iloc[0]["antecedent"] if len(churn_rules) > 0 else "Low engagement"

    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown(insight_box(
            f"🔗 <b>{n_rules} rules discovered</b> with lift ≥ {min_lift}. "
            f"High-lift rules reveal non-obvious cross-content affinities — "
            f"these are the bundling opportunities that a flat subscription catalogue misses."
        ), unsafe_allow_html=True)
    with i2:
        st.markdown(rec_box(
            f"📦 <b>Bundle recommendation:</b> Content types with lift &gt; 1.5 should be "
            f"surfaced together in the home feed and recommendation engine. "
            f"This increases time-on-platform and drives engagement score upward."
        ), unsafe_allow_html=True)
    with i3:
        st.markdown(insight_box(
            f"🚩 <b>Churn signal:</b> Rules where the consequent is "
            f"<b>Churn:Yes</b> identify antecedent behaviour patterns that predict cancellation. "
            f"Monitoring subscribers who match these antecedents enables proactive "
            f"retention outreach before the renewal decision."
        ), unsafe_allow_html=True)
