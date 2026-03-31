# theme.py  —  PitWall Analytics  ·  F1 Design System

# ── Palette ───────────────────────────────────────────────────────────────────
F1_RED     = "#E8002D"
F1_BLACK   = "#0A0A0A"
F1_WHITE   = "#F5F5F5"
F1_SILVER  = "#9B9B9B"
F1_GOLD    = "#FFD700"
F1_GREY    = "#1C1C1C"
F1_DGREY   = "#141414"

ACCENT_TEAL   = "#00C8E0"
ACCENT_GREEN  = "#2ECC8F"
ACCENT_AMBER  = "#FFC94D"
ACCENT_PURPLE = "#B57BEE"

# Pastel variants for gradients (no muddy mid-tones)
PASTEL_RED    = "#FF6B8A"
PASTEL_AMBER  = "#FFD580"
PASTEL_GREEN  = "#5EEAD4"
PASTEL_BLUE   = "#7DD3FC"
PASTEL_PURPLE = "#C4B5FD"

# ── Semantic colour mappings ───────────────────────────────────────────────────
PLAN_COLORS = {
    "Pit Lane":     F1_SILVER,
    "Podium":       F1_RED,
    "Paddock Club": F1_GOLD,
}

CHANNEL_COLORS = {
    "Paid Ad":      F1_RED,
    "Organic":      ACCENT_GREEN,
    "Social Media": ACCENT_AMBER,
    "Referral":     ACCENT_TEAL,
}

NPS_COLORS = {
    "Promoter":  ACCENT_GREEN,
    "Passive":   ACCENT_AMBER,
    "Detractor": F1_RED,
}

CHURN_COLORS = {
    "Active":  ACCENT_GREEN,
    "Churned": F1_RED,
}

RISK_COLORS = {
    "Low Risk":    ACCENT_GREEN,
    "Medium Risk": ACCENT_AMBER,
    "High Risk":   F1_RED,
}

SEGMENT_COLORS = {
    "Champions": "#FFD700",
    "Engaged":   "#5EEAD4",
    "At Risk":   "#FFC94D",
    "Dormant":   "#FF6B8A",
}


def hex_to_rgba(hex_color: str, alpha: float = 0.25) -> str:
    """Convert a #RRGGBB hex colour to rgba() — safe for Plotly fillcolor."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def base_layout(title: str = "", height: int = 400) -> dict:
    return dict(
        title=dict(
            text=title,
            font=dict(color=F1_WHITE, size=14, family="Arial Black, sans-serif"),
            x=0.01,
            pad=dict(b=4),
        ),
        paper_bgcolor=F1_GREY,
        plot_bgcolor=F1_DGREY,
        font=dict(color=F1_SILVER, family="Arial, sans-serif", size=12),
        height=height,
        margin=dict(l=48, r=24, t=52, b=44),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=F1_SILVER, size=11)),
        xaxis=dict(gridcolor="#252525", linecolor="#333", zerolinecolor="#333",
                   tickfont=dict(color=F1_SILVER)),
        yaxis=dict(gridcolor="#252525", linecolor="#333", zerolinecolor="#333",
                   tickfont=dict(color=F1_SILVER)),
        hoverlabel=dict(bgcolor=F1_GREY, bordercolor="#444",
                        font=dict(color=F1_WHITE, size=12)),
    )


F1_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Titillium Web', Arial, sans-serif !important;
    background-color: #0A0A0A !important;
    color: #F5F5F5;
}
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main, .main .block-container,
[data-testid="stTabsContent"] {
    background-color: #0A0A0A !important;
}
[data-testid="stSidebar"] {
    background-color: #0F0F0F !important;
    border-right: 2px solid #E8002D;
}
[data-testid="stTabs"] > div:first-child { border-bottom: 1px solid #2A2A2A; }
[data-testid="stTabs"] button {
    font-family: 'Titillium Web', sans-serif !important;
    font-weight: 700 !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    color: #555 !important;
    padding: 10px 20px !important;
    border-bottom: 3px solid transparent !important;
    background: transparent !important;
    text-transform: uppercase;
}
[data-testid="stTabs"] button:hover { color: #999 !important; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #E8002D !important;
    border-bottom: 3px solid #E8002D !important;
}
[data-testid="stTabsContent"] { padding-top: 1.5rem; }
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1A1A1A 0%, #111 100%) !important;
    border: 1px solid #222 !important;
    border-top: 3px solid #E8002D !important;
    border-radius: 6px !important;
    padding: 16px 20px !important;
}
[data-testid="metric-container"] label {
    color: #555 !important;
    font-size: 10px !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
}
[data-testid="stMetricValue"] {
    color: #F5F5F5 !important;
    font-size: 26px !important;
    font-weight: 900 !important;
}
[data-testid="stMetricDelta"] span { font-size: 11px !important; }
h1 { color: #E8002D !important; font-weight: 900 !important; letter-spacing: 3px !important; }
h2 { color: #F5F5F5 !important; font-weight: 700 !important;
     border-bottom: 1px solid #222; padding-bottom: 8px; margin-top: 0.5rem; }
h3 { color: #888 !important; font-weight: 600 !important; font-size: 0.85rem !important; }
hr { border-color: #1E1E1E !important; margin: 1.4rem 0 !important; }
[data-testid="stDataFrame"] { border: 1px solid #222; border-radius: 6px; overflow: hidden; }
.insight-box {
    background: linear-gradient(135deg, #1A1A1A, #111);
    border-left: 4px solid #E8002D;
    border-radius: 0 6px 6px 0;
    padding: 14px 18px;
    margin: 8px 0 14px 0;
    font-size: 13px;
    line-height: 1.75;
    color: #C8C8C8;
}
.insight-box b { color: #E8002D; }
.rec-box {
    background: linear-gradient(135deg, #0C1A0C, #0A0A0A);
    border-left: 4px solid #06D6A0;
    border-radius: 0 6px 6px 0;
    padding: 14px 18px;
    margin: 8px 0 14px 0;
    font-size: 13px;
    line-height: 1.75;
    color: #C8C8C8;
}
.rec-box b { color: #06D6A0; }
.warn-box {
    background: linear-gradient(135deg, #1A1400, #0A0A0A);
    border-left: 4px solid #FFB703;
    border-radius: 0 6px 6px 0;
    padding: 14px 18px;
    margin: 8px 0 14px 0;
    font-size: 13px;
    line-height: 1.75;
    color: #C8C8C8;
}
.warn-box b { color: #FFB703; }
.section-label {
    font-size: 9px;
    letter-spacing: 3.5px;
    text-transform: uppercase;
    color: #E8002D;
    font-weight: 700;
    margin-bottom: 6px;
    font-family: 'Titillium Web', sans-serif;
}
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stHeader"] { background-color: #0A0A0A !important; }
</style>
"""


def section_label(text: str) -> str:
    return f'<div class="section-label">{text}</div>'

def insight_box(html: str) -> str:
    return f'<div class="insight-box">{html}</div>'

def rec_box(html: str) -> str:
    return f'<div class="rec-box">{html}</div>'

def warn_box(html: str) -> str:
    return f'<div class="warn-box">{html}</div>'
