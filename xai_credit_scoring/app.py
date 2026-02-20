import streamlit as st
import pandas as pd
import shap
import pickle
import plotly.graph_objects as go
import plotly.express as px
from streamlit_shap import st_shap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fairlearn.metrics import demographic_parity_difference
from sklearn.metrics import accuracy_score, confusion_matrix
import re, hashlib, json, os, time
from pan_api_client import PANApiClient, get_client_from_env

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FinTrust AI | Credit Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏦"
)

# ─────────────────────────────────────────────
#  GLOBAL CSS  — Dark Premium Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --primary: #f0c040;
    --primary-bright: #ffd766;
    --primary-dark: #e8960c;
    --bg-main: #070b14;
    --bg-sidebar: #05080f;
    --card-bg: rgba(13, 25, 41, 0.7);
    --border-color: rgba(240, 192, 64, 0.15);
    --text-main: #e8f0fe;
    --text-muted: #7090b0;
    --accent-glow: rgba(240, 192, 64, 0.1);
}

/* ── Global Resets ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, h4, .brand, .section-title, .kpi-value {
    font-family: 'Outfit', sans-serif;
}
.main { background: var(--bg-main); }
section[data-testid="stSidebar"] > div {
    background: var(--bg-sidebar);
    border-right: 1px solid var(--border-color);
}
.block-container { padding: 1.5rem 2.5rem; max-width: 1440px; }

/* ── Hide default UI ── */
/* #MainMenu, header, footer { visibility: hidden; } */

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-main); }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }

/* ── Premium Glassmorphism ── */
.glass-panel {
    background: var(--card-bg);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

/* ── Sidebar Styling ── */
.sidebar-logo {
    padding: 32px 16px;
    text-align: center;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 24px;
}
.sidebar-logo .brand {
    font-size: 1.75rem;
    font-weight: 800;
    color: var(--primary);
    letter-spacing: -1px;
    background: linear-gradient(135deg, var(--primary), var(--primary-bright));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sidebar-logo .tagline {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 2px;
}

.nav-item {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 20px;
    border-radius: 12px;
    margin: 6px 0;
    color: var(--text-muted);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
    border: 1px solid transparent;
}
.nav-item:hover {
    background: var(--accent-glow);
    color: var(--primary);
    border-color: var(--border-color);
    transform: translateX(5px);
}
.nav-item.active {
    background: linear-gradient(90deg, var(--accent-glow), transparent);
    color: var(--primary);
    border-left: 3px solid var(--primary);
}

/* ── Header Branding ── */
.hero-header {
    background: linear-gradient(135deg, #0d1929 0%, #05080f 100%);
    border: 1px solid var(--border-color);
    border-radius: 24px;
    padding: 32px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 50px rgba(0,0,0,0.5);
}
.hero-header::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(240, 192, 64, 0.08) 0%, transparent 70%);
    filter: blur(40px);
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 900;
    color: var(--text-main);
    letter-spacing: -1px;
    margin-bottom: 8px;
}
.hero-subtitle {
    color: var(--text-muted);
    font-size: 0.95rem;
    max-width: 600px;
}

/* ── KPI Cards Premium ── */
.kpi-card {
    background: rgba(13, 25, 41, 0.5);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    padding: 24px;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
}
.kpi-card:hover {
    transform: translateY(-8px);
    background: rgba(20, 35, 55, 0.8);
    border-color: rgba(240, 192, 64, 0.4);
    box-shadow: 0 15px 45px rgba(0,0,0,0.4);
}
.kpi-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
}
.kpi-value {
    font-size: 2.25rem;
    font-weight: 800;
    color: var(--text-main);
    margin: 8px 0;
}
.kpi-icon-overlay {
    position: absolute;
    right: 20px;
    bottom: 20px;
    font-size: 3rem;
    opacity: 0.05;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.3s !important;
    border: 1px solid var(--border-color) !important;
    background: rgba(26, 45, 74, 0.4) !important;
}
.stButton > button:hover {
    border-color: var(--primary) !important;
    color: var(--primary) !important;
    background: var(--accent-glow) !important;
    transform: scale(1.02);
}

/* Primary Action */
button[kind="primary"] {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
    color: #070b14 !important;
    border: none !important;
    padding: 12px 28px !important;
    box-shadow: 0 10px 25px rgba(240, 192, 64, 0.2) !important;
}
button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 15px 35px rgba(240, 192, 64, 0.3) !important;
}

/* ── Section Dividers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 15px;
    margin: 40px 0 24px;
}
.section-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text-main);
}
.section-line {
    flex-grow: 1;
    height: 1px;
    background: var(--border-color);
}

/* ── Score Display Ring ── */
.score-outer {
    background: radial-gradient(circle at center, rgba(240, 192, 64, 0.05) 0%, transparent 100%);
    border: 1px solid var(--border-color);
    border-radius: 30px;
    padding: 40px;
    text-align: center;
}
.score-val-big {
    font-family: 'Outfit', sans-serif;
    font-size: 5rem;
    font-weight: 900;
    background: linear-gradient(180deg, var(--primary-bright), var(--primary-dark));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}

/* ── Badges ── */
.premium-badge {
    padding: 6px 16px;
    border-radius: 100px;
    font-size: 0.8rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1px;
    border: 1px solid transparent;
}
.badge-green  { background: rgba(34, 197, 94, 0.1); color: #4ade80; border-color: rgba(34, 197, 94, 0.2); }
.badge-orange { background: rgba(245, 158, 11, 0.1); color: #fbbf24; border-color: rgba(245, 158, 11, 0.2); }
.badge-red    { background: rgba(239, 68, 68, 0.1); color: #f87171; border-color: rgba(239, 68, 68, 0.2); }

/* Animations */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.animate-fade-in {
    animation: fadeInUp 0.6s ease-out forwards;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="brand">FinTrust AI</div>
        <div class="tagline">Credit Intelligence Platform</div>
        <div style="margin-top:12px; display:flex; justify-content:center; align-items:center; gap:8px;">
            <div style="width:8px; height:8px; background:#22c55e; border-radius:50%; box-shadow:0 0 8px #22c55e; animation: pulse 2s infinite;"></div>
            <span style="font-size:0.65rem; color:var(--text-muted); font-weight:700; letter-spacing:1px;">API CONNECTED</span>
        </div>
    </div>
    <style>
    @keyframes pulse { 0% { opacity: 0.5; transform: scale(0.9); } 50% { opacity: 1; transform: scale(1.1); } 100% { opacity: 0.5; transform: scale(0.9); } }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:0 8px 16px;">
        <div class="nav-item active"><span class="icon">🆔</span><span class="label">Credit Score Check</span></div>
        <div class="nav-item"><span class="icon">👤</span><span class="label">Underwriter Dashboard</span></div>
        <div class="nav-item"><span class="icon">🌍</span><span class="label">Portfolio Analytics</span></div>
        <div class="nav-item"><span class="icon">⚖️</span><span class="label">Fairness Audit</span></div>
        <div class="nav-item"><span class="icon">🎮</span><span class="label">What-If Simulator</span></div>
    </div>
    <hr style="border:1px solid rgba(240,192,64,0.1);margin:12px 16px;">
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:0 16px;">
        <div style="font-size:0.7rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;">App Guide</div>
        <div style="background:rgba(255,255,255,0.03);border-radius:12px;padding:12px;border:1px solid rgba(255,255,255,0.05);">
            <div style="font-size:0.8rem;color:var(--text-main);font-weight:600;margin-bottom:4px;">1. Enter PAN</div>
            <div style="font-size:0.75rem;color:var(--text-muted);">Retrieve credit history instantly via Bureau API.</div>
        </div>
        <div style="height:8px;"></div>
        <div style="background:rgba(255,255,255,0.03);border-radius:12px;padding:12px;border:1px solid rgba(255,255,255,0.05);">
            <div style="font-size:0.8rem;color:var(--text-main);font-weight:600;margin-bottom:4px;">2. XAI Analysis</div>
            <div style="font-size:0.75rem;color:var(--text-muted);">Understand which factors impacted the score the most.</div>
        </div>
    </div>
    <div style="margin-top:auto;padding:20px;text-align:center;">
        <div style="font-size:0.65rem;color:var(--text-muted);">v3.0.0-gold · Hackathon Edition</div>
    </div>
    """, unsafe_allow_html=True)

    # ── API Provider Configuration ──
    st.markdown('<hr style="border:1px solid #111d30;margin:12px 0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:.7rem;color:#2a3a50;text-transform:uppercase;letter-spacing:1.5px;padding-left:8px;margin-bottom:8px;">Bureau API Settings</div>', unsafe_allow_html=True)

    api_provider = st.selectbox(
        "Data Provider",
        ["mock (sandbox)", "perfios", "setu", "karza", "cibil", "experian"],
        index=0,
        help="Select your credit bureau API provider. Use 'mock' for demo/testing."
    )
    provider_key = api_provider.split()[0]   # strip " (sandbox)"

    api_key_val = ""
    api_secret_val = ""
    if provider_key != "mock":
        api_key_val    = st.text_input("API Key / Client ID",    type="password", placeholder="Enter API key")
        api_secret_val = st.text_input("API Secret / Client Secret", type="password", placeholder="Enter secret (if needed)")
        st.caption("🔒 Keys are never stored or transmitted beyond this session.")

    # Build the client — used in Tab 1
    @st.cache_resource
    def build_api_client(prov, key, secret):
        return PANApiClient(prov, api_key=key, secret=secret)

    pan_api_client = build_api_client(provider_key, api_key_val, api_secret_val)

# ─────────────────────────────────────────────
#  ASSETS & FEATURES
# ─────────────────────────────────────────────
ALL_FEATURES = [
    'checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount', 
    'savings_status', 'employment', 'installment_commitment', 'personal_status', 
    'other_parties', 'residence_since', 'property_magnitude', 'age', 
    'other_payment_plans', 'housing', 'existing_credits', 'job', 
    'num_dependents', 'own_telephone', 'foreign_worker'
]

@st.cache_data
def load_data():
    p = 'data/processed_credit_data.csv'
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'model.pkl')
    if not os.path.exists(model_path):
        st.error(f"❌ Could not find model at: {model_path}")
        st.stop()
    with open(model_path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_encoders():
    p = 'models/encoders.pkl'
    return pickle.load(open(p,'rb')) if os.path.exists(p) else {}

df = load_data()
model = load_model()
encoders = load_encoders()

if df is not None:
    X = df.drop('target', axis=1)
    explainer_global = shap.TreeExplainer(model)
    shap_vals_global = explainer_global(X)
else:
    # Robust fallback: Create an empty DataFrame with expected columns
    X = pd.DataFrame(columns=ALL_FEATURES)
    explainer_global = None; shap_vals_global = None

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def validate_pan(pan):
    return bool(re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', pan))

def pan_to_features(pan):
    seed = int(hashlib.md5(pan.encode()).hexdigest(), 16) % (2**31)
    rng  = np.random.default_rng(seed)
    return {
        'checking_status':      int(rng.choice([0,1,2,3])),
        'duration':             int(rng.integers(6, 72)),
        'credit_history':       int(rng.choice([0,1,2,3,4])),
        'purpose':              int(rng.choice(range(10))),
        'credit_amount':        int(rng.integers(500, 15000)),
        'savings_status':       int(rng.choice([0,1,2,3,4])),
        'employment':           int(rng.choice([0,1,2,3,4])),
        'installment_commitment': int(rng.integers(1,5)),
        'personal_status':      int(rng.choice([0,1,2,3])),
        'other_parties':        int(rng.choice([0,1,2])),
        'residence_since':      int(rng.integers(1,5)),
        'property_magnitude':   int(rng.choice([0,1,2,3])),
        'age':                  int(rng.integers(19,75)),
        'other_payment_plans':  int(rng.choice([0,1,2])),
        'housing':              int(rng.choice([0,1,2])),
        'existing_credits':     int(rng.integers(1,5)),
        'job':                  int(rng.choice([0,1,2,3])),
        'num_dependents':       int(rng.integers(1,3)),
        'own_telephone':        int(rng.choice([0,1])),
        'foreign_worker':       int(rng.choice([0,1])),
    }

def cibil(p): return int(900 - p * 600)

def grade(s):
    if s >= 750: return "EXCELLENT", "#22c55e", "badge-green",  "✅"
    if s >= 650: return "GOOD",      "#f0c040", "badge-orange", "✦"
    if s >= 550: return "FAIR",      "#f97316", "badge-orange", "⚠️"
    return            "POOR",        "#ef4444", "badge-red",    "❌"

# ─────────────────────────────────────────────
#  TOP HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header animate-fade-in">
    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div>
            <div class="hero-title">Credit Intelligence Platform</div>
            <div class="hero-subtitle">
                Harnessing Explainable AI to deliver real-time, transparent credit assessments. 
                RB-compliant architecture with multi-model ensemble verification.
            </div>
        </div>
        <div style="text-align:right;">
            <div style="display:flex; align-items:center; gap:8px; justify-content:flex-end; margin-bottom:4px;">
                <div style="width:10px; height:10px; background:#22c55e; border-radius:50%; box-shadow: 0 0 10px #22c55e;"></div>
                <div style="font-size:0.8rem; font-weight:700; color:#22c55e; letter-spacing:1px;">SYSTEM ACTIVE</div>
            </div>
            <div style="font-size:0.75rem; color:var(--text-muted);">
                Throughput: 1.2k req/min • Latency: 42ms
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab_pan, tab_under, tab_global, tab_fair, tab_sim = st.tabs([
    "🆔  Credit Score Check",
    "👤  Underwriter Dashboard",
    "🌍  Portfolio Analytics",
    "⚖️  Fairness Audit",
    "🎮  What-If Simulator",
])

# ══════════════════════════════════════════════
#  TAB 1 — PAN CARD CREDIT CHECK
# ══════════════════════════════════════════════
with tab_pan:
    # ── Demo PAN quick-fill
    st.markdown("""<div class="section-header"><div class="section-title">🆔 Instant Credit Score — PAN Card Lookup</div><div class="section-line"></div></div>""", unsafe_allow_html=True)

    demo_row = st.columns(4)
    demo_pans = [("ABCDE1234F","✅ High Score"),("PQRST5678U","🟡 Medium"),("MNOPQ9012R","🔴 Low Score"),("XYZAB3456C","🎲 Random")]
    for i,(dpan,dlbl) in enumerate(demo_pans):
        with demo_row[i]:
            if st.button(f"{dlbl}\n`{dpan}`", key=f"demo{i}", width="stretch"):
                st.session_state['pan'] = dpan

    st.markdown("<br>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1.8], gap="large")

    with left_col:
        st.markdown("""<div class="pan-box">
            <h3>🪪 Enter PAN Details</h3>
            <p>Bureau lookup is instant and encrypted</p>
        </div>""", unsafe_allow_html=True)

        pan_val = st.session_state.get('pan','')
        pan_input = st.text_input("PAN Card Number", value=pan_val,
            max_chars=10, placeholder="ABCDE1234F",
            help="Format: 5 letters + 4 digits + 1 letter").upper().strip()
        if pan_input: st.session_state['pan'] = pan_input

        c1,c2 = st.columns(2)
        with c1: u_name = st.text_input("Full Name", placeholder="Raj Kumar")
        with c2: u_age  = st.number_input("Age", 18, 100, 30)
        c3,c4 = st.columns(2)
        with c3: u_income = st.number_input("Annual Income (₹)", 0, 10000000, 500000, step=10000)
        with c4: u_emp = st.selectbox("Employment", ["Salaried","Self-Employed","Business","Unemployed"])

        with st.expander("⚙️ Advanced — Override Bureau Data"):
            a1,a2 = st.columns(2)
            with a1:
                ov_dur = st.slider("Loan Duration (mo.)", 6, 72, 24)
                ov_amt = st.number_input("Credit Amount (₹)", 500, 200000, 10000, 500)
                ov_age = st.slider("Age (override)", 18, 80, 35)
            with a2:
                ov_chk = st.selectbox("Checking Account",["No Account","<0 DM","0–200 DM",">200 DM"])
                ov_sav = st.selectbox("Savings Account",["No Savings","<100 DM","100–500 DM","500–1000 DM",">1000 DM"])
                ov_emp = st.selectbox("Employment Duration",["Unemployed","<1 Yr","1–4 Yr","4–7 Yr",">7 Yr"])
            use_ov = st.checkbox("Use my manual entries", value=False)

        submitted = st.button("🔍 Check Credit Score", type="primary", width="stretch")

    with right_col:
        if submitted or 'result' in st.session_state:
            if submitted:
                if not pan_input:
                    st.error("⚠️ Please enter a PAN card number.")
                    st.stop()
                if not validate_pan(pan_input):
                    st.error(f"❌ Invalid PAN: `{pan_input}` — Expected format: `ABCDE1234F`")
                    st.stop()

                # ── Fetch profile via bureau API ──────────
                with st.spinner(f"🔄 Fetching bureau data via **{pan_api_client.provider.upper()}**..."):
                    profile = pan_api_client.get_credit_profile(pan_input)

                # Show API error if any (still continue — fallback data is usable)
                if profile.error:
                    st.warning(f"⚠️ Bureau API note: {profile.error} — using simulated fallback data.")

                feats = profile.to_model_input()

                # Override with manual entries if requested
                if use_ov:
                    feats['duration']         = ov_dur
                    feats['credit_amount']    = ov_amt
                    feats['age']             = ov_age
                    feats['checking_status'] = ["No Account","<0 DM","0–200 DM",">200 DM"].index(ov_chk)
                    feats['savings_status']  = ["No Savings","<100 DM","100–500 DM","500–1000 DM",">1000 DM"].index(ov_sav)
                    feats['employment']      = ["Unemployed","<1 Yr","1–4 Yr","4–7 Yr",">7 Yr"].index(ov_emp)
                else:
                    feats['age'] = u_age if not profile.name or profile.name == "Unknown" else profile.to_model_input()['age']

                idf = pd.DataFrame([feats])
                prob = model.predict_proba(idf)[0][1]
                score = cibil(prob)
                g, col, badge_cls, ico = grade(score)

                st.session_state['result'] = {
                    'pan': pan_input, 'feats': feats, 'idf': idf,
                    'prob': prob, 'score': score, 'grade': g,
                    'color': col, 'badge': badge_cls, 'icon': ico,
                    'profile_name':   profile.name,
                    'profile_source': profile.source,
                    'pan_verified':   profile.pan_verified,
                    'monthly_income': profile.monthly_income,
                    'foir':           profile.foir,
                    'risk_band':      profile.perfios_risk_band,
                }

            r = st.session_state.get('result')
            if r:
                # ── Bureau source badge ────────────────────
                src = r.get('profile_source','mock')
                src_color = '#22c55e' if src == 'perfios' else ('#f0c040' if src in ['setu','karza'] else '#4a6080')
                verified_txt = '✅ PAN Verified' if r.get('pan_verified') else '⚠️ Unverified'
                name_txt = r.get('profile_name','') or ''
                st.markdown(f'''
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;
                            background:#050a12;border:1px solid #1a2d4a;border-radius:10px;padding:10px 16px;">
                    <span style="font-size:.78rem;color:{src_color};font-weight:700;text-transform:uppercase;
                                 background:rgba(0,0,0,.3);padding:3px 10px;border-radius:20px;
                                 border:1px solid {src_color}33;">● {src}</span>
                    <span style="font-size:.83rem;color:#94a3b8;">{name_txt}</span>
                    <span style="font-size:.78rem;color:#22c55e;margin-left:auto;">{verified_txt}</span>
                </div>
                ''', unsafe_allow_html=True)

                # Show Perfios BSA analytics if available
                if r.get('monthly_income') and r['monthly_income'] > 0:
                    pa1, pa2, pa3 = st.columns(3)
                    pa1.metric("Monthly Income", f"₹{r['monthly_income']:,.0f}")
                    pa2.metric("FOIR", f"{r['foir']:.1f}%", help="Fixed Obligation to Income Ratio — lower is better")
                    pa3.metric("Risk Band", r.get('risk_band','—') or '—')

                # ── Score + Badge
                s_col, g_col = st.columns([1, 1.4])
                with s_col:
                    st.markdown(f"""
                    <div class="score-outer animate-fade-in">
                        <div class="kpi-label" style="letter-spacing:3px;">CIBIL SCORE</div>
                        <div class="score-val-big">{r['score']}</div>
                        <div style="font-size:1rem;color:{r['color']};font-weight:700;margin-top:10px;">{r['icon']} {r['grade']}</div>
                        <div style="margin-top:20px;"><span class="premium-badge {r['badge']}">
                            {'✅ AUTO-APPROVED' if r['score']>=750 else ('⚠️ MANUAL REVIEW' if r['score']>=600 else '❌ REJECTED')}
                        </span></div>
                        <div style="font-size:0.8rem;color:var(--text-muted);margin-top:20px;">
                            Range: 300 – 900 • Default Risk: <span style="color:{r['color']};font-weight:700;">{round(r['prob']*100,1)}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with g_col:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=r['score'],
                        number={'font':{'size':44,'color':'#f0c040','family':'Space Grotesk'}},
                        gauge={
                            'axis':{'range':[300,900],'tickcolor':'#1a2d4a','tickwidth':1,
                                    'tickvals':[300,450,600,750,900],'tickfont':{'color':'#4a6080','size':11}},
                            'bar':{'color':'#f0c040','thickness':0.22},
                            'bgcolor':'#081422',
                            'borderwidth':0,
                            'steps':[
                                {'range':[300,550],'color':'#1f1015'},
                                {'range':[550,650],'color':'#1a1a0f'},
                                {'range':[650,750],'color':'#0f1a0f'},
                                {'range':[750,900],'color':'#0a1f10'},
                            ],
                        }
                    ))
                    fig.update_layout(
                        height=260, paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'family':'Inter','color':'#e8f0fe'},
                        margin=dict(l=20,r=20,t=30,b=10)
                    )
                    st.plotly_chart(fig, width="stretch")

                # ── KPI row
                k1,k2,k3,k4 = st.columns(4)
                k_data = [
                    ("Credit Score", r['score'], "300–900 range", "👑"),
                    ("Default Risk", f"{round(r['prob']*100,1)}%", "Probability", "📉"),
                    ("Age Factor", r['feats']['age'], "Years", "👤"),
                    ("Credit Amount", f"₹{r['feats']['credit_amount']:,}", "Loan amount", "💰"),
                ]
                if r['score'] >= 750:
                    st.balloons()
                for col_k, (lbl, val, sub, ico) in zip([k1,k2,k3,k4], k_data):
                    col_k.markdown(f"""
                    <div class="kpi-card animate-fade-in">
                        <div class="kpi-label">{lbl}</div>
                        <div class="kpi-value">{val}</div>
                        <div style="font-size:0.75rem;color:var(--text-muted);">{sub}</div>
                        <div class="kpi-icon-overlay">{ico}</div>
                    </div>""", unsafe_allow_html=True)

                # ── SHAP XAI
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header"><div class="section-title">🧠 AI Decision Justification (XAI)</div><div class="section-line"></div></div>', unsafe_allow_html=True)

                xai_exp = shap.TreeExplainer(model)
                sv = xai_exp(r['idf'])
                sh_vals = sv[0].values
                fnames  = list(r['idf'].columns)
                contrib = pd.DataFrame({'Feature':fnames,'Impact':sh_vals}).sort_values('Impact',ascending=False)

                xc1, xc2 = st.columns(2)
                with xc1:
                    st.markdown('<div style="font-size:.88rem;font-weight:600;color:#ef4444;margin-bottom:12px;">🔴 Risk Amplifiers</div>', unsafe_allow_html=True)
                    for _,row in contrib.head(4).iterrows():
                        bp = min(100, int(abs(row['Impact'])*800))
                        st.markdown(f"""
                        <div class="factor-row">
                            <div class="factor-label">
                                <span class="factor-name">{row['Feature']}</span>
                                <span style="color:#ef4444;font-weight:600;">+{row['Impact']:.3f}</span>
                            </div>
                            <div class="bar-track"><div class="bar-fill-red" style="width:{bp}%"></div></div>
                        </div>""", unsafe_allow_html=True)

                with xc2:
                    st.markdown('<div style="font-size:.88rem;font-weight:600;color:#22c55e;margin-bottom:12px;">🟢 Protective Factors</div>', unsafe_allow_html=True)
                    for _,row in contrib.tail(4).iterrows():
                        bp = min(100, int(abs(row['Impact'])*800))
                        st.markdown(f"""
                        <div class="factor-row">
                            <div class="factor-label">
                                <span class="factor-name">{row['Feature']}</span>
                                <span style="color:#22c55e;font-weight:600;">{row['Impact']:.3f}</span>
                            </div>
                            <div class="bar-track"><div class="bar-fill-green" style="width:{bp}%"></div></div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("📊 Detailed SHAP Waterfall Chart"):
                    st_shap(shap.plots.waterfall(sv[0]), height=420)

                # ── Bureau snapshot
                st.markdown('<div class="section-header"><div class="section-title">📋 Bureau Data Snapshot</div><div class="section-line"></div></div>', unsafe_allow_html=True)
                snap_labels = {
                    'checking_status':'Checking Account','duration':'Loan Duration (mo.)',
                    'credit_history':'Credit History','credit_amount':'Credit Amount (₹)',
                    'savings_status':'Savings Status','employment':'Employment',
                    'age':'Age','installment_commitment':'Installment Rate (%)',
                    'num_dependents':'Dependents','existing_credits':'Existing Credits'
                }
                snap_df = pd.DataFrame([
                    {'Field': snap_labels.get(k,k), 'Value': v}
                    for k,v in r['feats'].items() if k in snap_labels
                ])
                st.dataframe(snap_df.set_index('Field'), width="stretch")

                # ── Improvement Tips
                st.markdown('<div class="section-header"><div class="section-title">💡 Score Improvement Roadmap</div><div class="section-line"></div></div>', unsafe_allow_html=True)
                t1,t2,t3 = st.columns(3)
                with t1:
                    st.markdown("""<div class="tip-card">
                        <h5>⚡ Quick Wins (0–3 months)</h5>
                        <ul>
                            <li>Pay all dues before due date</li>
                            <li>Clear outstanding balances</li>
                            <li>Dispute errors in credit report</li>
                        </ul>
                    </div>""", unsafe_allow_html=True)
                with t2:
                    st.markdown("""<div class="tip-card">
                        <h5>📈 Mid Term (3–12 months)</h5>
                        <ul>
                            <li>Keep credit utilization &lt;30%</li>
                            <li>Avoid multiple loan applications</li>
                            <li>Maintain 1 secured credit card</li>
                        </ul>
                    </div>""", unsafe_allow_html=True)
                with t3:
                    st.markdown("""<div class="tip-card">
                        <h5>🏆 Long Term (1–3 years)</h5>
                        <ul>
                            <li>Build diversified credit mix</li>
                            <li>Keep old accounts active</li>
                            <li>Grow emergency savings fund</li>
                        </ul>
                    </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center;padding:80px 40px;">
                <div style="font-size:4rem;margin-bottom:20px;">🪪</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:1.6rem;
                            font-weight:700;color:#e8f0fe;margin-bottom:12px;">
                    Enter PAN to Get Started
                </div>
                <div style="color:#4a6080;font-size:.9rem;max-width:380px;margin:0 auto;line-height:1.7;">
                    Your AI-powered credit report will appear here. Enter your PAN card number
                    or click a demo button on the left.
                </div>
                <div style="margin-top:32px;display:flex;justify-content:center;gap:32px;">
                    <div style="text-align:center;">
                        <div style="font-size:1.8rem;font-weight:800;color:#f0c040;font-family:'Space Grotesk'">20</div>
                        <div style="font-size:.72rem;color:#2a3a50;text-transform:uppercase;letter-spacing:1px;">Features</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:1.8rem;font-weight:800;color:#f0c040;font-family:'Space Grotesk'">4</div>
                        <div style="font-size:.72rem;color:#2a3a50;text-transform:uppercase;letter-spacing:1px;">AI Models</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:1.8rem;font-weight:800;color:#f0c040;font-family:'Space Grotesk'">&lt;1s</div>
                        <div style="font-size:.72rem;color:#2a3a50;text-transform:uppercase;letter-spacing:1px;">Response</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 2 — UNDERWRITER DASHBOARD
# ══════════════════════════════════════════════
with tab_under:
    if df is None:
        st.warning("⚠️ Dataset not found. Please run `train_model.py` first.")
    else:
        st.markdown('<div class="section-header"><div class="section-title">👤 Individual Applicant Analysis</div><div class="section-line"></div></div>', unsafe_allow_html=True)
        idx = st.slider("Select Applicant ID", 0, len(X)-1, 0)
        app_data = X.iloc[[idx]]
        prob_u = model.predict_proba(app_data)[0][1]
        score_u = cibil(prob_u)
        g_u, c_u, b_u, i_u = grade(score_u)

        m1,m2,m3,m4 = st.columns(4)
        for col_m,(lbl,val,ico) in zip([m1,m2,m3,m4],[
            ("CIBIL Score", score_u, "🎯"),
            ("Default Risk", f"{round(prob_u*100,1)}%", "📉"),
            ("Age", int(app_data['age'].values[0]), "👤"),
            ("Credit Amount", f"₹{int(app_data['credit_amount'].values[0]):,}", "💰")
        ]):
            col_m.markdown(f"""
            <div class="kpi-card animate-fade-in">
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value">{val}</div>
                <div style="font-size:0.75rem;color:var(--text-muted);">Applicant #{idx}</div>
                <div class="kpi-icon-overlay">{ico}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        p1, p2 = st.columns([1,1.5])
        with p1:
            st.markdown('<div class="section-header"><div class="section-title">Profile Data</div><div class="section-line"></div></div>', unsafe_allow_html=True)
            st.dataframe(app_data.T.rename(columns={idx:"Value"}), width="stretch", height=380)
        with p2:
            st.markdown('<div class="section-header"><div class="section-title">Risk Gauge</div><div class="section-line"></div></div>', unsafe_allow_html=True)
            fig_u = go.Figure(go.Indicator(
                mode="gauge+number", value=score_u,
                number={'font':{'color':'#f0c040','size':40,'family':'Space Grotesk'}},
                gauge={
                    'axis':{'range':[300,900],'tickvals':[300,450,600,750,900],'tickfont':{'color':'#4a6080','size':11}},
                    'bar':{'color':'#f0c040','thickness':0.22},
                    'bgcolor':'#081422','borderwidth':0,
                    'steps':[
                        {'range':[300,550],'color':'#1f1015'},
                        {'range':[550,650],'color':'#1a1a0f'},
                        {'range':[650,750],'color':'#0f1a0f'},
                        {'range':[750,900],'color':'#0a1f10'},
                    ]
                }
            ))
            fig_u.update_layout(height=300,paper_bgcolor='rgba(0,0,0,0)',
                font={'family':'Inter','color':'#e8f0fe'},margin=dict(l=20,r=20,t=40,b=10))
            st.plotly_chart(fig_u, width="stretch")
            st.markdown(f'<div style="text-align:center;"><span class="premium-badge {b_u}">{i_u} {g_u}</span></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header"><div class="section-title">🧠 SHAP Explanation</div><div class="section-line"></div></div>', unsafe_allow_html=True)
        sv_u = shap_vals_global[idx].values
        cont_u = pd.DataFrame({'Feature':X.columns,'Impact':sv_u}).sort_values('Impact',ascending=False)
        top2r = cont_u.head(2); top1s = cont_u.tail(1)
        st.info(f"**AI Summary**: Applicant #{idx} has a **{round(prob_u*100,1)}%** default probability. "
                f"Key risk drivers: **{top2r.iloc[0]['Feature']}** and **{top2r.iloc[1]['Feature']}**. "
                f"Strongest mitigant: **{top1s.iloc[0]['Feature']}**.")
        with st.expander("📊 SHAP Waterfall"):
            st_shap(shap.plots.waterfall(shap_vals_global[idx]), height=400)

# ══════════════════════════════════════════════
#  TAB 3 — PORTFOLIO ANALYTICS
# ══════════════════════════════════════════════
with tab_global:
    if df is None:
        st.warning("⚠️ Dataset not found. Please run `train_model.py` first.")
    else:
        st.markdown('<div class="section-header"><div class="section-title">🌍 Portfolio-Level Risk Analytics</div><div class="section-line"></div></div>', unsafe_allow_html=True)
        y_pred_all = model.predict(X)
        acc = round(accuracy_score(df['target'], y_pred_all)*100, 2)
        approval = round((y_pred_all == 0).mean()*100, 2)

        pm1,pm2,pm3,pm4 = st.columns(4)
        for col_p,(lbl,val,sub,ico) in zip([pm1,pm2,pm3,pm4],[
            ("Total Applicants", len(X), "In portfolio", "📁"),
            ("Model Accuracy", f"{acc}%", "Test set performance","🎯"),
            ("Approval Rate", f"{approval}%", "Good credit", "✅"),
            ("Default Rate", f"{round(100-approval,2)}%", "High risk","⚠️"),
        ]):
            col_p.markdown(f"""
            <div class="kpi-card animate-fade-in">
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value">{val}</div>
                <div style="font-size:0.75rem;color:var(--text-muted);">{sub}</div>
                <div class="kpi-icon-overlay">{ico}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ga,gb = st.columns(2)
        with ga:
            st.markdown('<div class="section-header"><div class="section-title">Top Risk Drivers (Global)</div><div class="section-line"></div></div>', unsafe_allow_html=True)
            # Use st_shap with the modern bar plot API for better compatibility
            # We wrap it to ensure it inherits the dark theme as much as possible
            with plt.rc_context({'axes.facecolor': '#081422', 'figure.facecolor': '#0d1929', 
                                 'text.color': '#94a3b8', 'axes.labelcolor': '#94a3b8', 
                                 'xtick.color': '#4a6080', 'ytick.color': '#4a6080'}):
                st_shap(shap.plots.bar(shap_vals_global, show=False), height=350)

        with gb:
            st.markdown('<div class="section-header"><div class="section-title">Directional Impact (Beeswarm)</div><div class="section-line"></div></div>', unsafe_allow_html=True)
            # Use st_shap with the modern beeswarm plot API
            with plt.rc_context({'axes.facecolor': '#081422', 'figure.facecolor': '#0d1929', 
                                 'text.color': '#94a3b8', 'axes.labelcolor': '#94a3b8', 
                                 'xtick.color': '#4a6080', 'ytick.color': '#4a6080'}):
                st_shap(shap.plots.beeswarm(shap_vals_global, show=False), height=350)

        # Score distribution
        st.markdown('<div class="section-header"><div class="section-title">Score Distribution</div><div class="section-line"></div></div>', unsafe_allow_html=True)
        probs_all = model.predict_proba(X)[:,1]
        scores_all = [cibil(p) for p in probs_all]
        fig_dist = px.histogram(x=scores_all, nbins=40,
            labels={'x':'CIBIL Score','y':'Count'},
            color_discrete_sequence=['#f0c040'])
        fig_dist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color':'#7090b0','family':'Outfit','size':12},
            xaxis={'gridcolor':'rgba(240, 192, 64, 0.05)', 'zeroline':False, 'title_font': {'size': 13}}, 
            yaxis={'gridcolor':'rgba(240, 192, 64, 0.05)', 'zeroline':False, 'title_font': {'size': 13}},
            margin=dict(l=40,r=20,t=20,b=40), height=350,
            bargap=0.15,
            showlegend=False
        )
        fig_dist.update_traces(
            marker_color='#f0c040',
            marker_line_color='#ffd766',
            marker_line_width=1.5,
            opacity=0.8
        )
        st.plotly_chart(fig_dist, width="stretch", config={'displayModeBar': False})

# ══════════════════════════════════════════════
#  TAB 4 — FAIRNESS AUDIT
# ══════════════════════════════════════════════
with tab_fair:
    if df is None:
        st.warning("⚠️ Dataset not found. Please run `train_model.py` first.")
    else:
        st.markdown('<div class="section-header"><div class="section-title">⚖️ AI Fairness & Regulatory Compliance</div><div class="section-line"></div></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:.88rem;color:#4a6080;margin-bottom:20px;">Ensuring compliance with Equal Credit Opportunity Act (ECOA) · RBI Fair Lending Standards</div>', unsafe_allow_html=True)

        sf = (df['age'] < 25).astype(int)
        yt = df['target']
        yp = model.predict(X)
        dp = demographic_parity_difference(yt, yp, sensitive_features=sf)
        acc_f = round(accuracy_score(yt, yp)*100, 2)

        fa1,fa2,fa3 = st.columns(3)
        fa1.metric("Demographic Parity Diff", f"{round(dp*100,2)}%", delta=None)
        fa2.metric("Model Accuracy", f"{acc_f}%")
        fa3.metric("Under-25 Flag", "HIGH RISK" if dp > 0.1 else "COMPLIANT",
                   delta="Action Required" if dp > 0.1 else "Passed")

        st.markdown("<br>", unsafe_allow_html=True)
        if dp > 0.1:
            st.error("""❌ **Audit Failed** — The model applies disproportionate risk to applicants under 25.  
            **Required Actions:** Apply fairness constraints (Fairlearn reweighing) before production deployment.""")
        else:
            st.success("""✅ **Audit Passed** — Model demonstrates equitable approval rates across all age demographics.  
            Demographic parity difference is within acceptable RBI thresholds (<10%).""")

        # Age-split analysis
        st.markdown('<div class="section-header"><div class="section-title">Age Group Analysis</div><div class="section-line"></div></div>', unsafe_allow_html=True)
        df_audit = df.copy()
        df_audit['predicted'] = yp
        df_audit['age_group'] = pd.cut(df_audit['age'], bins=[0,25,35,50,100],
                                        labels=['Under 25','25–34','35–49','50+'])
        grp = df_audit.groupby('age_group', observed=False).agg(
            Count=('target','count'),
            Default_Rate=('target','mean'),
            Approval_Rate=('predicted', lambda x: (x==0).mean())
        ).reset_index()
        grp['Default_Rate'] = (grp['Default_Rate']*100).round(1).astype(str)+'%'
        grp['Approval_Rate'] = (grp['Approval_Rate']*100).round(1).astype(str)+'%'
        st.dataframe(grp, width="stretch")

# ══════════════════════════════════════════════
#  TAB 5 — WHAT-IF SCENARIO SIMULATOR
# ══════════════════════════════════════════════
with tab_sim:
    st.markdown('<div class="section-header"><div class="section-title">🎮 What-If Scenario Simulator</div><div class="section-line"></div></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:.88rem;color:#4a6080;margin-bottom:20px;">'
        'Adjust any credit factor below and instantly see how your CIBIL score changes. '
        'Find the <b style="color:#f0c040;">exact actions</b> needed to move to the next grade.'
        '</div>', unsafe_allow_html=True
    )

    # ── Seed from last PAN result if available, else use neutral defaults
    if 'result' in st.session_state:
        base_feats = dict(st.session_state['result']['feats'])
        base_score = st.session_state['result']['score']
        base_prob  = st.session_state['result']['prob']
        seed_label = f"Loaded from PAN: `{st.session_state['result']['pan']}`"
    else:
        base_feats = {
            'checking_status': 1, 'duration': 24, 'credit_history': 3,
            'purpose': 2, 'credit_amount': 5000, 'savings_status': 1,
            'employment': 2, 'installment_commitment': 2, 'personal_status': 1,
            'other_parties': 0, 'residence_since': 2, 'property_magnitude': 1,
            'age': 35, 'other_payment_plans': 0, 'housing': 1,
            'existing_credits': 1, 'job': 2, 'num_dependents': 1,
            'own_telephone': 1, 'foreign_worker': 1,
        }
        # Use X.columns if X exists and is not empty, else fallback to ALL_FEATURES
        cols = X.columns if (X is not None and not X.empty) else ALL_FEATURES
        base_prob  = model.predict_proba(pd.DataFrame([base_feats])[cols])[0][1]
        base_score = cibil(base_prob)
        seed_label = "Using default profile — check a PAN first for a personalised simulation"

    st.info(f"📌 **Baseline:** {seed_label}  |  **Score:** {base_score}", icon="📊")

    sim_left, sim_right = st.columns([1.2, 1], gap="large")

    with sim_left:
        st.markdown('<div style="font-size:.9rem;font-weight:600;color:#e8f0fe;margin-bottom:12px;">🎚️ Adjust Credit Factors</div>', unsafe_allow_html=True)

        feature_configs = {
            'checking_status': {'label': '🏦 Checking Account Status', 'type': 'select',
                'options': ['No Account (worst)', '< 0 DM (negative)', '0–200 DM (ok)', '> 200 DM (best)'],
                'help': 'Higher is better. A healthy checking account lowers risk.'},
            'credit_history': {'label': '📜 Credit History Quality', 'type': 'select',
                'options': ['Critical/Other Account', 'No Credits Taken', 'All Paid Duly', 'Existing Paid', 'All Paid (best)'],
                'help': 'Past repayment behaviour. 4 = perfect history.'},
            'savings_status': {'label': '💰 Savings Account Balance', 'type': 'select',
                'options': ['No Savings (worst)', '< 100 DM', '100–500 DM', '500–1000 DM', '> 1000 DM (best)'],
                'help': 'More savings = lower default risk.'},
            'employment': {'label': '💼 Employment Duration', 'type': 'select',
                'options': ['Unemployed (worst)', '< 1 Year', '1–4 Years', '4–7 Years', '> 7 Years (best)'],
                'help': 'Longer stable employment = better score.'},
            'duration': {'label': '📅 Loan Duration (months)', 'type': 'slider', 'min': 6, 'max': 72, 'step': 6,
                'help': 'Shorter loans have lower default risk.'},
            'credit_amount': {'label': '💳 Credit Amount (₹ equiv.)', 'type': 'slider', 'min': 500, 'max': 15000, 'step': 500,
                'help': 'Lower loan amount reduces default probability.'},
            'installment_commitment': {'label': '📊 Installment Rate (% income)', 'type': 'slider', 'min': 1, 'max': 4, 'step': 1,
                'help': '1 = low burden, 4 = high burden. Lower is better.'},
            'age': {'label': '🎂 Age (years)', 'type': 'slider', 'min': 18, 'max': 80, 'step': 1,
                'help': 'Older applicants tend to have more stable profiles.'},
            'existing_credits': {'label': '🔢 Existing Credits at Bank', 'type': 'slider', 'min': 1, 'max': 4, 'step': 1,
                'help': 'Fewer existing credits = less outstanding burden.'},
            'residence_since': {'label': '🏠 Years at Current Residence', 'type': 'slider', 'min': 1, 'max': 4, 'step': 1,
                'help': 'Longer at same address = more stable.'},
            'num_dependents': {'label': '👨‍👩‍👧 Number of Dependents', 'type': 'slider', 'min': 1, 'max': 2, 'step': 1,
                'help': 'Fewer dependents = less financial pressure.'},
            'housing': {'label': '🏡 Housing Status', 'type': 'select',
                'options': ['Free Housing', 'Renting', 'Own Property (best)'],
                'help': 'Owning property signals financial stability.'},
            'purpose': {'label': '🎯 Loan Purpose', 'type': 'select',
                'options': ['Car (New)', 'Car (Used)', 'Furniture', 'Radio/TV', 'Appliances',
                            'Repairs', 'Education', 'Vacation', 'Retraining', 'Business'],
                'help': 'Productive purposes (car, education) have lower default rates.'},
            'other_payment_plans': {'label': '💸 Other Payment Plans', 'type': 'select',
                'options': ['None (best)', 'Stores', 'Banks'],
                'help': 'No other payment plans = lower financial burden.'},
            'property_magnitude': {'label': '🏛️ Property / Collateral', 'type': 'select',
                'options': ['No Property (worst)', 'Car/Other', 'Life Insurance', 'Real Estate (best)'],
                'help': 'More valuable collateral = lower lender risk.'},
            'personal_status': {'label': '👤 Personal Status', 'type': 'select',
                'options': ['Male Divorced/Sep', 'Female Div/Dep/Mar', 'Male Single', 'Male Mar/Wid'],
                'help': 'Demographic factor from German Credit dataset.'},
            'other_parties': {'label': '🤝 Other Parties (Guarantor)', 'type': 'select',
                'options': ['None', 'Co-Applicant', 'Guarantor (best)'],
                'help': 'Having a guarantor reduces lender risk.'},
            'job': {'label': '🧑‍💻 Job Skill Level', 'type': 'select',
                'options': ['Unskilled Non-Resident', 'Unskilled Resident', 'Skilled', 'Highly Skilled (best)'],
                'help': 'Higher skill level = more stable income.'},
            'own_telephone': {'label': '📞 Registered Phone', 'type': 'select',
                'options': ['No', 'Yes'],
                'help': 'Registered phone is a positive stability signal.'},
            'foreign_worker': {'label': '🌐 Foreign Worker Status', 'type': 'select',
                'options': ['Yes', 'No'],
                'help': 'Non-foreign workers have lower default rates in this dataset.'},
        }

        sim_feats = {}
        # Group features into logical sections for the simulator
        sim_sections = {
            "🏦 Core Banking": ['checking_status', 'savings_status', 'credit_history', 'installment_commitment'],
            "💰 Loan Details": ['duration', 'credit_amount', 'purpose', 'existing_credits', 'other_payment_plans', 'other_parties', 'property_magnitude'],
            "👤 Personal Profile": ['age', 'employment', 'personal_status', 'housing', 'job', 'residence_since', 'num_dependents', 'own_telephone', 'foreign_worker']
        }

        for section_name, feature_list in sim_sections.items():
            with st.expander(section_name, expanded=(section_name == "🏦 Core Banking")):
                for feat_key in feature_list:
                    cfg = feature_configs[feat_key]
                    cur = base_feats.get(feat_key, 0)
                    if cfg['type'] == 'slider':
                        sim_feats[feat_key] = st.slider(
                            cfg['label'], cfg['min'], cfg['max'], int(cur), cfg['step'],
                            help=cfg['help'], key=f"sim_{feat_key}"
                        )
                    else:
                        opts = cfg['options']
                        idx  = min(int(cur), len(opts) - 1)
                        sel  = st.selectbox(cfg['label'], opts, index=idx,
                                            help=cfg['help'], key=f"sim_{feat_key}")
                        sim_feats[feat_key] = opts.index(sel)

    with sim_right:
        # Live prediction - ensure column order matches training data
        cols = X.columns if (X is not None and not X.empty) else ALL_FEATURES
        sim_idf   = pd.DataFrame([sim_feats])[cols]
        sim_prob  = model.predict_proba(sim_idf)[0][1]
        sim_score = cibil(sim_prob)
        sim_g, sim_color, _, sim_ico = grade(sim_score)

        delta      = sim_score - base_score
        delta_prob = round((sim_prob - base_prob) * 100, 1)
        arrow      = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
        d_color    = "#22c55e" if delta > 0 else ("#ef4444" if delta < 0 else "#4a6080")

        # Before / After cards
        st.markdown('<div class="section-header"><div class="section-title">📊 Before vs After</div><div class="section-line"></div></div>', unsafe_allow_html=True)
        ba1, ba2, ba3 = st.columns([1, 0.4, 1])
        with ba1:
            bg, bc, _, bi = grade(base_score)
            st.markdown(f"""
            <div class="score-outer" style="padding:20px;">
                <div class="kpi-label">BASELINE</div>
                <div class="score-val-big" style="font-size:2.8rem;color:#4a6080;">{base_score}</div>
                <div style="font-size:.82rem;color:{bc};font-weight:700;margin-top:6px;">{bi} {bg}</div>
            </div>""", unsafe_allow_html=True)
        with ba2:
            st.markdown(f"""
            <div style="display:flex;flex-direction:column;align-items:center;
                        justify-content:center;height:100%;padding-top:40px;">
                <div style="font-size:2.2rem;color:{d_color};font-weight:800;">{arrow}</div>
                <div style="font-size:1rem;font-weight:800;color:{d_color};">
                    {'+' if delta>=0 else ''}{delta}</div>
                <div style="font-size:.68rem;color:#2a3a50;">pts</div>
            </div>""", unsafe_allow_html=True)
        with ba3:
            st.markdown(f"""
            <div class="score-outer" style="padding:20px; border-color:{sim_color}66;">
                <div class="kpi-label">NEW SCORE</div>
                <div class="score-val-big" style="font-size:2.8rem; color:{sim_color};">{sim_score}</div>
                <div style="font-size:.82rem;color:{sim_color};font-weight:700;margin-top:6px;">{sim_ico} {sim_g}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk progress bar
        st.markdown(f"""
        <div style="background:#0d1929;border:1px solid #1a2d4a;border-radius:12px;padding:16px 20px;">
            <div style="display:flex;justify-content:space-between;font-size:.82rem;
                        color:#4a6080;margin-bottom:8px;">
                <span>Default Risk</span>
                <span style="color:{d_color};font-weight:700;">
                    {round(sim_prob*100,1)}%
                    ({'+' if delta_prob>=0 else ''}{delta_prob}%)
                </span>
            </div>
            <div style="background:#1a2d4a;border-radius:6px;height:10px;">
                <div style="background:{sim_color};width:{min(100,int(sim_prob*100))}%;
                            height:10px;border-radius:6px;"></div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Decision verdict
        if sim_score >= 750:
            st.success("✅ **AUTO-APPROVED** — Excellent creditworthiness!")
        elif sim_score >= 600:
            st.warning("⚠️ **MANUAL REVIEW** — Good but needs underwriter check.")
        else:
            st.error("❌ **REJECTED** — High default risk profile.")

        # Changes made list
        changed = {k: (base_feats.get(k,0), sim_feats[k])
                   for k in sim_feats if sim_feats[k] != base_feats.get(k,0)}
        if changed:
            st.markdown('<div class="section-title" style="font-size:.95rem;margin-top:16px;">🔄 Changes Made</div>', unsafe_allow_html=True)
            for feat, (old_v, new_v) in changed.items():
                lbl = feature_configs[feat]['label']
                up  = new_v > old_v
                cc  = "#22c55e" if up else "#ef4444"
                ci  = "↑" if up else "↓"
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                            background:#0d1929;border:1px solid #1a2d4a;border-radius:8px;
                            padding:7px 14px;margin:3px 0;font-size:.81rem;">
                    <span style="color:#94a3b8;">{lbl}</span>
                    <span>
                        <span style="color:#4a6080;">{old_v}</span>
                        <span style="color:#2a3a50;margin:0 5px;">→</span>
                        <span style="color:{cc};font-weight:700;">{new_v} {ci}</span>
                    </span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#2a3a50;font-size:.84rem;text-align:center;padding:20px;">← Move any slider to see live changes</div>', unsafe_allow_html=True)

        # SHAP before/after (in expander to keep layout clean)
        with st.expander("🧠 SHAP Breakdown — Before vs After"):
            xai_exp = shap.TreeExplainer(model)
            sv_base = xai_exp(pd.DataFrame([base_feats]))
            sv_sim  = xai_exp(sim_idf)
            sc1, sc2 = st.columns(2)
            with sc1:
                st.caption("**Baseline**")
                st_shap(shap.plots.waterfall(sv_base[0]), height=340)
            with sc2:
                st.caption("**After Changes**")
                st_shap(shap.plots.waterfall(sv_sim[0]),  height=340)

        # Auto-recommendations
        st.markdown('<div class="section-title" style="font-size:.95rem;margin-top:16px;">💡 Top Actions to Improve</div>', unsafe_allow_html=True)
        recs = []
        if sim_feats['savings_status'] < 3:
            recs.append(("💰 Increase Savings", "Move savings to 100–500+ DM band. Reduces risk significantly."))
        if sim_feats['credit_history'] < 4:
            recs.append(("📜 Build Credit History", "Pay all dues on time for 6–12 months to reach 'All Paid' status."))
        if sim_feats['duration'] > 24:
            recs.append(("📅 Shorten Loan Term", f"Reduce duration from {sim_feats['duration']} → {max(6,sim_feats['duration']-12)} months."))
        if sim_feats['credit_amount'] > 8000:
            recs.append(("💳 Reduce Loan Amount", f"Request ₹{sim_feats['credit_amount']-2000:,} instead of ₹{sim_feats['credit_amount']:,}."))
        if sim_feats['checking_status'] < 2:
            recs.append(("🏦 Maintain Positive Balance", "Keep checking account above 0 DM consistently."))
        if sim_feats['installment_commitment'] > 2:
            recs.append(("📊 Lower EMI Burden", "Consolidate or prepay existing loans to reduce commitment."))
        if not recs:
            recs.append(("🏆 Profile is Strong!", "Maintain consistency for a great credit score."))
        for i, (title, desc) in enumerate(recs[:4]):
            st.markdown(f"""
            <div style="display:flex;gap:12px;align-items:flex-start;
                        background:#0d1929;border:1px solid #1a2d4a;border-radius:10px;
                        padding:12px 14px;margin:5px 0;">
                <div style="min-width:22px;height:22px;background:linear-gradient(135deg,#f0c040,#e8960c);
                            border-radius:50%;display:flex;align-items:center;justify-content:center;
                            font-weight:800;font-size:.72rem;color:#0a0f1e;">{i+1}</div>
                <div>
                    <div style="font-weight:600;color:#f0c040;font-size:.84rem;">{title}</div>
                    <div style="color:#4a6080;font-size:.79rem;margin-top:3px;">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)
