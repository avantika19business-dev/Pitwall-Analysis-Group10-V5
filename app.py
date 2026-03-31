"""
app.py  —  PitWall Analytics Dashboard  (Group Assignment Version)
──────────────────────────────────────────────────────────────────
6-tab structure required by the group assignment rubric:
  1. Executive Overview   (Descriptive + Diagnostic)
  2. Classification       (all 6 algorithms + comparison)
  3. Clustering           (KMeans, elbow, PCA, persona cards)
  4. Regression           (Linear, Ridge, Lasso)
  5. Association Rules    (Apriori basket mining)
  6. Prescriptive         (uplift, A/B sim, CLV, recommendations)
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent
for p in [str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

st.set_page_config(
    page_title   = "PitWall Analytics",
    page_icon    = "🏎",
    layout       = "wide",
    initial_sidebar_state = "collapsed",
)

from theme          import F1_CSS
from data_generator import load_data
import tab1_descriptive
import tab2_diagnostic
import tab_classification
import tab_clustering
import tab_regression
import tab_association
import tab4_prescriptive

st.markdown(F1_CSS, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def _load():
    return load_data()

with st.spinner("🏎  Loading race data…"):
    subs, sess, mrr = _load()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(90deg, #0A0A0A 0%, #1A0005 50%, #0A0A0A 100%);
    border-bottom: 3px solid #E8002D;
    padding: 20px 28px 16px 28px;
    margin: -1rem -1rem 1.5rem -1rem;
">
    <div style="
        font-size: 9px; letter-spacing: 4px; color: #E8002D; font-weight: 700;
        text-transform: uppercase; margin-bottom: 6px;
        font-family: 'Titillium Web', Arial, sans-serif;
    ">
        F1 PERFORMANCE DATA PLATFORM &nbsp;·&nbsp; SUBSCRIBER ANALYTICS &nbsp;·&nbsp; GROUP ASSIGNMENT
    </div>
    <div style="
        font-size: 28px; font-weight: 900; color: #F5F5F5; letter-spacing: 1.5px;
        font-family: 'Titillium Web', Arial Black, sans-serif;
    ">
        🏎&nbsp; PITWALL ANALYTICS
    </div>
    <div style="
        font-size: 12px; color: #666; margin-top: 5px;
        font-family: 'Titillium Web', Arial, sans-serif; letter-spacing: 0.5px;
    ">
        800 Subscribers &nbsp;·&nbsp; 29,240 Sessions &nbsp;·&nbsp;
        3 Plan Tiers &nbsp;·&nbsp; Seasons 2023–2024 &nbsp;·&nbsp;
        6 Classifiers &nbsp;·&nbsp; KMeans Clustering &nbsp;·&nbsp;
        3 Regression Models &nbsp;·&nbsp; Association Rule Mining
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "📋  Overview",
    "🔍  Diagnostic",
    "🎯  Classification",
    "🔵  Clustering",
    "📉  Regression",
    "🔗  Association Rules",
    "🚀  Prescriptive",
])

with t1:
    tab1_descriptive.render(subs, sess, mrr)

with t2:
    tab2_diagnostic.render(subs, sess, mrr)

with t3:
    tab_classification.render(subs, sess, mrr)

with t4:
    tab_clustering.render(subs, sess, mrr)

with t5:
    tab_regression.render(subs, sess, mrr)

with t6:
    tab_association.render(subs, sess, mrr)

with t7:
    tab4_prescriptive.render(subs, sess, mrr)
