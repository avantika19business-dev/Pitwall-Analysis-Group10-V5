"""
tab4_prescriptive.py  —  Prescriptive Analytics
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from theme import (
    base_layout, section_label, insight_box, rec_box, warn_box, hex_to_rgba,
    F1_RED, F1_WHITE, F1_SILVER, F1_GREY, F1_DGREY, F1_GOLD,
    ACCENT_TEAL, ACCENT_GREEN, ACCENT_AMBER,
    PLAN_COLORS, SEGMENT_COLORS, RISK_COLORS,
)
from model_utils import engineer_features, train_churn_model, segment_customers


@st.cache_data(show_spinner=False)
def _get_scored(subs: pd.DataFrame, sess: pd.DataFrame) -> pd.DataFrame:
    df  = engineer_features(subs, sess)
    out = train_churn_model(df)
    _, _, _, _, _, _, _, _, df = out
    df  = segment_customers(df)

    rng = np.random.default_rng(42)
    n   = len(df)

    df["treatment_response"] = np.clip(
        df["churn_prob"] * 0.65
        + (df["avg_engagement"] / 100) * 0.35
        + rng.normal(0, 0.07, n),
        0, 1,
    )
    df["control_response"] = np.clip(
        1 - df["churn_prob"] + rng.normal(0, 0.05, n),
        0, 1,
    )

    def _uplift(row):
        t, c = row["treatment_response"], row["control_response"]
        if   t > 0.5 and c < 0.5: return "Persuadable"
        elif t > 0.5 and c > 0.5: return "Sure Thing"
        elif t < 0.5 and c < 0.5: return "Lost Cause"
        else:                      return "Sleeping Dog"

    df["uplift_segment"]   = df.apply(_uplift, axis=1)
    df["priority_score"]   = (df["churn_prob"] * df["Monthly Price Usd"]).round(2)
    df["predicted_clv_6m"] = (df["Monthly Price Usd"] * 6 * (1 - df["churn_prob"])).round(2)
    return df


_UPLIFT_COLORS = {
    "Persuadable":  ACCENT_GREEN,
    "Sure Thing":   ACCENT_TEAL,
    "Lost Cause":   F1_RED,
    "Sleeping Dog": F1_SILVER,
}


def render(subs: pd.DataFrame, sess: pd.DataFrame, mrr: pd.DataFrame) -> None:
    st.markdown("## Prescriptive — *What Should We Do?*")
    st.markdown(
        "Uplift modelling, A/B test simulator, "
        "CLV forecasting, and data-backed strategic recommendations."
    )
    st.markdown("---")

    with st.spinner("Building prescriptive models…"):
        df = _get_scored(subs, sess)

    active       = df[df["churn_flag"] == 0].copy()
    persuadables = active[active["uplift_segment"] == "Persuadable"]

    # ── Uplift 4-quadrant ─────────────────────────────────────────────────────
    st.markdown(section_label("UPLIFT MODEL — WHO RESPONDS TO RETENTION OFFERS?"), unsafe_allow_html=True)
    st.markdown(
        "Simulates which active subscribers will respond to a discount/offer "
        "vs those who stay or leave regardless of intervention."
    )
    col1, col2 = st.columns([2, 1])

    with col1:
        samp = active.sample(min(500, len(active)), random_state=42)
        fig1 = px.scatter(
            samp,
            x="control_response",
            y="treatment_response",
            color="uplift_segment",
            color_discrete_map=_UPLIFT_COLORS,
            size="Monthly Price Usd",
            size_max=16,
            hover_data={
                "Subscriber Id": True,
                "Plan": True,
                "churn_prob": ":.1%",
                "Monthly Price Usd": True,
            },
            labels={
                "control_response":   "Control Response (no offer)",
                "treatment_response": "Treatment Response (with offer)",
            },
        )
        fig1.add_hline(y=0.5, line_dash="dash", line_color=F1_SILVER, line_width=1)
        fig1.add_vline(x=0.5, line_dash="dash", line_color=F1_SILVER, line_width=1)
        for txt, ax, ay, color in [
            ("🎯 PERSUADABLES",  0.25, 0.80, ACCENT_GREEN),
            ("✅ SURE THINGS",   0.75, 0.80, ACCENT_TEAL),
            ("❌ LOST CAUSES",   0.25, 0.20, F1_RED),
            ("😴 SLEEPING DOGS", 0.75, 0.20, F1_SILVER),
        ]:
            fig1.add_annotation(
                x=ax, y=ay, text=txt, showarrow=False,
                font=dict(color=color, size=11, family="Arial Black"),
            )
        fig1.update_layout(**base_layout(
            "Uplift 4-Quadrant: Retention Intervention Targeting", height=420))
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        seg_c = active["uplift_segment"].value_counts()
        fig2 = go.Figure(go.Bar(
            x=seg_c.values.tolist(),
            y=seg_c.index.tolist(),
            orientation="h",
            marker=dict(color=[_UPLIFT_COLORS.get(s, F1_SILVER) for s in seg_c.index]),
            text=seg_c.values.tolist(),
            textposition="outside",
            textfont=dict(color=F1_WHITE),
            hovertemplate="<b>%{y}</b><br>Subscribers: %{x}<extra></extra>",
        ))
        fig2.update_layout(**base_layout("Segment Counts", height=240))
        st.plotly_chart(fig2, use_container_width=True)

        avg_price_p = float(persuadables["Monthly Price Usd"].mean()) if len(persuadables) > 0 else 0
        st.markdown(insight_box(
            f"🎯 <b>{len(persuadables)} Persuadables</b> identified — the only group where a "
            f"discount or offer generates incremental lift. "
            f"Avg plan price: <b>${avg_price_p:.2f}/mo</b>."
        ), unsafe_allow_html=True)

    # ── A/B test simulator ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(section_label("A/B TEST SIMULATOR — DESIGN YOUR RETENTION CAMPAIGN"), unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    discount   = s1.slider("Discount Offered (%)", 5, 50, 20, 5)
    target     = s2.selectbox(
        "Target Group",
        ["Persuadables Only", "All At-Risk", "Pit Lane", "Podium", "Paddock Club"],
    )
    n_per_arm  = s3.slider("Sample Size / Arm", 30, 300, 100, 10)
    base_churn = s4.slider("Baseline Churn Rate (%)", 10, 60, 32, 1)

    _effect = {
        "Persuadables Only": 0.16, "All At-Risk": 0.08,
        "Pit Lane": 0.10, "Podium": 0.07, "Paddock Club": 0.05,
    }
    effect    = _effect.get(target, 0.08) * (discount / 20)
    treat_cr  = max(0.02, (base_churn / 100) - effect)
    ctrl_cr   = base_churn / 100

    rng2     = np.random.default_rng(99)
    ctrl_ch  = int(rng2.binomial(n_per_arm, ctrl_cr))
    treat_ch = int(rng2.binomial(n_per_arm, treat_cr))
    cont     = np.array([[ctrl_ch,  n_per_arm - ctrl_ch],
                          [treat_ch, n_per_arm - treat_ch]])
    _, p_val, _, _ = stats.chi2_contingency(cont)
    lift      = (ctrl_cr - treat_cr) / ctrl_cr * 100
    avg_price = {"Persuadables Only": 15, "All At-Risk": 12,
                 "Pit Lane": 9.99, "Podium": 19.99, "Paddock Club": 39.99}.get(target, 12)
    saved     = max(0, ctrl_ch - treat_ch)
    mrr_saved = saved * avg_price
    cost      = n_per_arm * avg_price * (discount / 100) * 0.5
    net_mrr   = mrr_saved - cost

    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Baseline Churn",    f"{ctrl_cr:.1%}")
    r2.metric("Treated Churn",     f"{treat_cr:.1%}", f"−{lift:.1f}% lift")
    r3.metric("Subscribers Saved", f"{saved}")
    r4.metric("Net MRR Benefit",   f"${net_mrr:,.0f}")
    r5.metric("p-value",           f"{p_val:.3f}",
              "✅ Significant" if p_val < 0.05 else "⚠️ Not Sig.")

    ab1, ab2 = st.columns(2)
    with ab1:
        fig_ab = go.Figure(go.Bar(
            x=["Control\nChurned", "Control\nRetained",
               "Treated\nChurned", "Treated\nRetained"],
            y=[ctrl_ch, n_per_arm - ctrl_ch, treat_ch, n_per_arm - treat_ch],
            marker_color=[F1_RED, ACCENT_GREEN, "#FF6B6B", "#06D6A0"],
            text=[ctrl_ch, n_per_arm - ctrl_ch, treat_ch, n_per_arm - treat_ch],
            textposition="auto",
            textfont=dict(color=F1_WHITE),
        ))
        p_color = ACCENT_GREEN if p_val < 0.05 else ACCENT_AMBER
        fig_ab.add_annotation(
            x=0.5, y=1.13, xref="paper", yref="paper", showarrow=False,
            text=(f"p = {p_val:.3f}  "
                  f"{'✅ Statistically Significant' if p_val < 0.05 else '⚠️ Not Significant at α = 0.05'}"),
            font=dict(color=p_color, size=12),
        )
        lo_ab = base_layout("A/B Test — Simulated Outcome", height=300)
        lo_ab["yaxis"]["title"] = "Subscribers"
        fig_ab.update_layout(**lo_ab)
        st.plotly_chart(fig_ab, use_container_width=True)

    with ab2:
        months = list(range(1, 7))
        fig_roi = go.Figure()
        fig_roi.add_trace(go.Scatter(
            x=months,
            y=[net_mrr * m for m in months],
            name="Cumulative Benefit",
            mode="lines+markers",
            line=dict(color=ACCENT_GREEN, width=2.5),
            fill="tozeroy",
            fillcolor=hex_to_rgba(ACCENT_GREEN, 0.08),
        ))
        fig_roi.add_trace(go.Scatter(
            x=months,
            y=[cost * 0.3 * m for m in months],
            name="Cumulative Cost",
            mode="lines+markers",
            line=dict(color=F1_RED, width=2),
        ))
        lo_roi = base_layout("Projected 6-Month Campaign ROI", height=300)
        lo_roi["xaxis"]["title"] = "Month"
        lo_roi["yaxis"]["title"] = "Cumulative USD"
        fig_roi.update_layout(**lo_roi)
        st.plotly_chart(fig_roi, use_container_width=True)

    # ── Priority intervention table ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(section_label("TOP 30 PRIORITY INTERVENTION TARGETS"), unsafe_allow_html=True)
    st.markdown(
        "Persuadable active subscribers ranked by Priority Score "
        "= Churn Probability × Monthly Price."
    )
    top30 = persuadables.sort_values("priority_score", ascending=False).head(30)
    if len(top30) == 0:
        top30 = active.sort_values("priority_score", ascending=False).head(30)

    disp = top30[[
        "Subscriber Id", "Plan", "Region", "Monthly Price Usd",
        "avg_engagement", "avg_duration", "churn_prob",
        "segment_label", "priority_score",
    ]].copy()
    disp.columns = [
        "Subscriber", "Plan", "Region", "Price ($)",
        "Avg Engagement", "Avg Duration (min)",
        "Churn Prob", "Segment", "Priority Score",
    ]
    disp["Churn Prob"]         = disp["Churn Prob"].map("{:.1%}".format)
    disp["Avg Engagement"]     = disp["Avg Engagement"].map("{:.1f}".format)
    disp["Avg Duration (min)"] = disp["Avg Duration (min)"].map("{:.1f}".format)
    disp["Priority Score"]     = disp["Priority Score"].map("{:.2f}".format)
    st.dataframe(disp, use_container_width=True, height=360)

    # ── Predicted CLV ─────────────────────────────────────────────────────────
    st.markdown("---")
    clv1, clv2 = st.columns(2)

    with clv1:
        st.markdown(section_label("PREDICTED 6-MONTH CLV — BY PLAN"), unsafe_allow_html=True)
        clv_plan = (
            active.groupby("Plan")["predicted_clv_6m"]
            .agg(total="sum", avg="mean")
            .reset_index()
        )
        fig_c1 = go.Figure(go.Bar(
            x=clv_plan["Plan"],
            y=clv_plan["total"],
            marker=dict(
                color=[PLAN_COLORS.get(p, F1_SILVER) for p in clv_plan["Plan"]],
                line=dict(color=F1_DGREY, width=1),
            ),
            text=[f"${v:,.0f}" for v in clv_plan["total"]],
            textposition="outside",
            textfont=dict(color=F1_WHITE),
            customdata=clv_plan["avg"].tolist(),
            hovertemplate=(
                "<b>%{x}</b><br>Total CLV: $%{y:,.0f}<br>"
                "Avg per sub: $%{customdata:.0f}<extra></extra>"
            ),
        ))
        lc1 = base_layout("Total Predicted 6-Month CLV by Plan", height=320)
        lc1["yaxis"]["title"] = "Predicted CLV (USD)"
        lc1["yaxis"]["range"] = [0, float(clv_plan["total"].max()) * 1.3]
        fig_c1.update_layout(**lc1)
        st.plotly_chart(fig_c1, use_container_width=True)

    with clv2:
        st.markdown(section_label("PREDICTED 6-MONTH CLV — BY SEGMENT"), unsafe_allow_html=True)
        clv_seg = (
            active.groupby("segment_label")["predicted_clv_6m"]
            .mean()
            .reset_index()
        )
        clv_seg.columns = ["segment", "avg_clv"]
        fig_c2 = go.Figure(go.Bar(
            x=clv_seg["segment"],
            y=clv_seg["avg_clv"],
            marker=dict(
                color=[SEGMENT_COLORS.get(s, F1_SILVER) for s in clv_seg["segment"]],
                line=dict(color=F1_DGREY, width=1),
            ),
            text=[f"${v:.0f}" for v in clv_seg["avg_clv"]],
            textposition="outside",
            textfont=dict(color=F1_WHITE),
            hovertemplate="<b>%{x}</b><br>Avg CLV: $%{y:.2f}<extra></extra>",
        ))
        lc2 = base_layout("Avg Predicted 6-Month CLV by Segment", height=320)
        lc2["yaxis"]["title"] = "Avg Predicted CLV (USD)"
        lc2["yaxis"]["range"] = [0, float(clv_seg["avg_clv"].max()) * 1.3]
        fig_c2.update_layout(**lc2)
        st.plotly_chart(fig_c2, use_container_width=True)

    # ── Strategic recommendations ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown(section_label("6 STRATEGIC RECOMMENDATIONS"), unsafe_allow_html=True)

    pit_churn  = float(subs[subs["Plan"] == "Pit Lane"]["churn_flag"].mean() * 100)
    padd_churn = float(subs[subs["Plan"] == "Paddock Club"]["churn_flag"].mean() * 100)
    best_ch    = subs.groupby("Acquisition Channel")["churn_flag"].mean().idxmin()
    top_reason = (subs[subs["Churned"] == "Yes"]["Churn Reason"].value_counts().idxmax())
    worst_reg  = subs.groupby("Region")["churn_flag"].mean().idxmax()
    dormant_n  = len(df[df["segment_label"] == "Dormant"])
    persuad_n  = len(persuadables)

    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown(rec_box(
            "<b>🏎 REC 1 — Upgrade Campaign: Pit Lane → Podium</b><br>"
            f"Pit Lane churns at {pit_churn:.1f}% vs {padd_churn:.1f}% for Paddock Club. "
            f"Offer Pit Lane Persuadables a '2 months free on Podium' trial. "
            f"Expected trade: short-term MRR dip, long-term retention gain and higher ARPU."
        ), unsafe_allow_html=True)

        st.markdown(rec_box(
            "<b>📅 REC 2 — 90-Day Onboarding Programme</b><br>"
            "Cohort retention curves show the steepest dropout in months 1–3. "
            "Implement a structured email series tied to each subscriber's top content type "
            "at signup. For subscribers with zero sessions in the first 7 days, trigger an "
            "in-app prompt surfacing the highest-engagement content category."
        ), unsafe_allow_html=True)

        st.markdown(rec_box(
            f"<b>📣 REC 3 — Reallocate CAC to {best_ch}</b><br>"
            f"'{best_ch}' delivers the lowest churn rate of any acquisition channel. "
            f"Shifting 20% of Paid Ad budget to referral incentives improves LTV:CAC "
            f"without requiring any product change."
        ), unsafe_allow_html=True)

    with rc2:
        st.markdown(warn_box(
            f"<b>🌍 REC 4 — Regional Pricing Pilot: {worst_reg}</b><br>"
            f"{worst_reg} shows the highest churn rate, concentrated in Pit Lane. "
            f"The #1 churn reason is '{top_reason}', which in price-sensitive markets "
            f"often signals a value mismatch. Test a 20% localised discount — it costs "
            f"less than the MRR lost to churn at current rates."
        ), unsafe_allow_html=True)

        st.markdown(warn_box(
            f"<b>🎮 REC 5 — Gamified Re-engagement for Dormant Segment</b><br>"
            f"{dormant_n} subscribers sit in the Dormant KMeans cluster — "
            f"low engagement, short sessions, elevated churn probability. "
            f"Launch a 'PitWall Race Week' challenge: weekly prediction contests tied to "
            f"live race data, targeted exclusively at this segment."
        ), unsafe_allow_html=True)

        st.markdown(rec_box(
            f"<b>💡 REC 6 — Proactive Content Roadmap Communication</b><br>"
            f"'{top_reason}' is the #1 stated churn reason. Publish a quarterly content "
            f"roadmap and distribute it 2 weeks before each subscriber's renewal date. "
            f"Run an A/B test on the {persuad_n} Persuadables first — "
            f"measure 30-day churn lift before full rollout."
        ), unsafe_allow_html=True)
