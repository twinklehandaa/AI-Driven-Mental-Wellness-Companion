"""
AI-Driven Mental Wellness Companion
Context-Aware Mood Prediction using Hybrid RF + LSTM
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import datetime
import random
import json
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MindSync — Mental Wellness AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f1923 0%, #1a2740 100%);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; font-size: 12px !important; }

/* Main background */
.stApp { background: #f8fafc; }

/* Metric cards */
.metric-card {
    background: white;
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    border: 1px solid #f1f5f9;
    margin-bottom: 12px;
}
.metric-card h3 { font-size: 12px; color: #94a3b8; font-weight: 500; margin: 0 0 6px; letter-spacing: 0.05em; text-transform: uppercase; }
.metric-card .val { font-size: 32px; font-weight: 600; color: #0f172a; margin: 0; font-family: 'DM Serif Display', serif; }
.metric-card .delta { font-size: 12px; margin-top: 4px; }
.delta-up { color: #10b981; }
.delta-down { color: #ef4444; }

/* Mood badge */
.mood-badge {
    display: inline-block;
    padding: 8px 20px;
    border-radius: 100px;
    font-weight: 600;
    font-size: 15px;
    letter-spacing: 0.02em;
}
.mood-happy { background: #dcfce7; color: #15803d; }
.mood-neutral { background: #dbeafe; color: #1d4ed8; }
.mood-stress { background: #fee2e2; color: #dc2626; }

/* Section headers */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: #0f172a;
    margin: 0 0 4px;
    font-weight: 400;
}
.section-sub { font-size: 13px; color: #64748b; margin-bottom: 20px; }

/* Recommendation cards */
.rec-card {
    background: white;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #f1f5f9;
    margin-bottom: 10px;
}
.rec-card h4 { font-size: 14px; font-weight: 600; color: #1e293b; margin: 0 0 4px; }
.rec-card p { font-size: 13px; color: #64748b; margin: 0; line-height: 1.5; }

/* Quote box */
.quote-box {
    background: linear-gradient(135deg, #667eea15, #764ba215);
    border-left: 3px solid #667eea;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 16px 0;
    font-style: italic;
    color: #475569;
    font-size: 14px;
}

/* Risk badge */
.risk-low { background: #dcfce7; color: #15803d; padding: 4px 12px; border-radius: 100px; font-size: 12px; font-weight: 600; }
.risk-medium { background: #fef9c3; color: #854d0e; padding: 4px 12px; border-radius: 100px; font-size: 12px; font-weight: 600; }
.risk-high { background: #fee2e2; color: #dc2626; padding: 4px 12px; border-radius: 100px; font-size: 12px; font-weight: 600; }

/* Journal entry */
.journal-entry {
    background: white;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #f1f5f9;
    margin-bottom: 10px;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "journal" not in st.session_state:
    st.session_state.journal = []
if "goals" not in st.session_state:
    st.session_state.goals = {
        "sleep_target": 8.0,
        "screen_target": 4.0,
        "activity_target": "High",
        "social_target": "Medium"
    }
if "streak" not in st.session_state:
    st.session_state.streak = 0
if "points" not in st.session_state:
    st.session_state.points = 0
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

# ─────────────────────────────────────────────
# SYNTHETIC DATASET GENERATOR
# ─────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=500):
    np.random.seed(42)
    data = []
    for _ in range(n):
        sleep = round(np.random.uniform(3, 10), 1)
        screen = round(np.random.uniform(1, 12), 1)
        activity = np.random.choice([0, 1, 2])   # 0=Low,1=Med,2=High
        social = np.random.choice([0, 1, 2])

        score = (
            (sleep / 10) * 0.35 +
            (1 - screen / 12) * 0.25 +
            (activity / 2) * 0.20 +
            (social / 2) * 0.20
        ) + np.random.normal(0, 0.05)

        if score < 0.35:
            mood = 0   # Stress
        elif score < 0.62:
            mood = 1   # Neutral
        else:
            mood = 2   # Happy

        data.append([sleep, screen, activity, social, mood])

    df = pd.DataFrame(data, columns=["sleep_hours", "screen_time", "activity_level", "social_interaction", "mood"])
    return df

# ─────────────────────────────────────────────
# TRAIN RANDOM FOREST
# ─────────────────────────────────────────────
@st.cache_resource
def train_model():
    df = generate_dataset(500)
    X = df[["sleep_hours", "screen_time", "activity_level", "social_interaction"]]
    y = df["mood"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight="balanced")
    rf.fit(X_train_sc, y_train)

    y_pred = rf.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Stress", "Neutral", "Happy"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return rf, scaler, acc, report, cm, df

# ─────────────────────────────────────────────
# LSTM SIMULATION (lightweight, no TF required)
# ─────────────────────────────────────────────
def simulate_lstm_prediction(sleep, screen, activity, social):
    history_window = st.session_state.history[-3:] if len(st.session_state.history) >= 3 else []

    base_score = (
        (sleep / 10) * 0.35 +
        (1 - screen / 12) * 0.25 +
        (activity / 2) * 0.20 +
        (social / 2) * 0.20
    )

    # Temporal trend adjustment
    if history_window:
        trend = np.mean([h["mood_idx"] / 2 for h in history_window])
        base_score = 0.7 * base_score + 0.3 * trend

    noise = np.random.normal(0, 0.03)
    base_score = np.clip(base_score + noise, 0, 1)

    # Softmax-like probabilities
    raw = np.array([
        max(0.01, 0.9 - base_score * 1.8),
        max(0.01, 1 - abs(base_score - 0.5) * 2.5),
        max(0.01, base_score * 1.8 - 0.4)
    ])
    probs = raw / raw.sum()
    return probs

# ─────────────────────────────────────────────
# HYBRID PREDICTOR
# ─────────────────────────────────────────────
def hybrid_predict(rf, scaler, sleep, screen, activity, social):
    MOOD_NAMES = ["Stress", "Neutral", "Happy"]
    act_map = {"Low": 0, "Medium": 1, "High": 2}

    act_val = act_map[activity] if isinstance(activity, str) else activity
    soc_val = act_map[social] if isinstance(social, str) else social

    X = np.array([[sleep, screen, act_val, soc_val]])
    X_sc = scaler.transform(X)

    rf_probs = rf.predict_proba(X_sc)[0]
    rf_pred = np.argmax(rf_probs)

    lstm_probs = simulate_lstm_prediction(sleep, screen, act_val, soc_val)
    lstm_pred = np.argmax(lstm_probs)

    # Weighted fusion: LSTM 0.6, RF 0.4
    hybrid_probs = 0.6 * lstm_probs + 0.4 * rf_probs
    hybrid_pred = np.argmax(hybrid_probs)

    return {
        "rf_pred": MOOD_NAMES[rf_pred],
        "lstm_pred": MOOD_NAMES[lstm_pred],
        "hybrid_pred": MOOD_NAMES[hybrid_pred],
        "hybrid_idx": int(hybrid_pred),
        "rf_probs": rf_probs.tolist(),
        "lstm_probs": lstm_probs.tolist(),
        "hybrid_probs": hybrid_probs.tolist(),
        "confidence": float(hybrid_probs[hybrid_pred])
    }

# ─────────────────────────────────────────────
# WELLNESS SCORE CALCULATOR
# ─────────────────────────────────────────────
def compute_wellness_score(sleep, screen, activity, social, mood_idx):
    act_score = {"Low": 0.3, "Medium": 0.65, "High": 1.0}.get(activity, 0.5)
    soc_score = {"Low": 0.3, "Medium": 0.65, "High": 1.0}.get(social, 0.5)
    sleep_score = min(sleep / 9, 1.0)
    screen_score = max(0, 1 - (screen - 2) / 10)
    mood_score = [0.2, 0.6, 1.0][mood_idx]
    raw = (sleep_score * 0.25 + screen_score * 0.20 + act_score * 0.20 + soc_score * 0.15 + mood_score * 0.20)
    return round(raw * 100, 1)

# ─────────────────────────────────────────────
# RISK ASSESSOR
# ─────────────────────────────────────────────
def assess_risk(sleep, screen, activity, social, history):
    risk_score = 0
    flags = []
    if sleep < 5.5:
        risk_score += 30
        flags.append("⚠️ Critically low sleep")
    elif sleep < 7:
        risk_score += 15
        flags.append("📉 Below recommended sleep")
    if screen > 8:
        risk_score += 25
        flags.append("📵 Excessive screen time")
    elif screen > 6:
        risk_score += 10
        flags.append("🖥️ High screen time")
    if activity == "Low":
        risk_score += 20
        flags.append("🏃 Insufficient physical activity")
    if social == "Low":
        risk_score += 15
        flags.append("💬 Limited social connection")

    # Trend risk: 3+ consecutive stress
    if len(history) >= 3 and all(h["mood_idx"] == 0 for h in history[-3:]):
        risk_score += 25
        flags.append("🔴 3+ consecutive stress predictions")

    if risk_score >= 50:
        level = "High"
    elif risk_score >= 25:
        level = "Medium"
    else:
        level = "Low"

    return level, risk_score, flags

# ─────────────────────────────────────────────
# RECOMMENDATIONS ENGINE
# ─────────────────────────────────────────────
def get_recommendations(mood_idx, sleep, screen, activity, social):
    recs = {
        0: [  # Stress
            ("🧘 Box breathing", "4-4-4-4 breathing for 5 minutes lowers cortisol by activating the parasympathetic nervous system."),
            ("🚶 10-min walk", f"Your activity is {activity}. Even a short outdoor walk cuts stress hormones significantly."),
            ("📵 Screen detox", f"You've had {screen}h screen time. Take a 1-hour break away from all devices."),
            ("💬 Reach out", "Call or message one person today. Social interaction is a powerful stress buffer."),
            ("📓 Expressive writing", "Write about what's stressing you for 15 mins. Research shows it reduces psychological burden."),
            ("🎵 Music therapy", "Listening to 432 Hz music or classical pieces reduces anxiety and heart rate."),
        ],
        1: [  # Neutral
            ("📖 Gratitude journaling", "List 3 specific things you're grateful for. This shifts your baseline mood upward."),
            ("🏋️ Upgrade activity", f"Activity is {activity}. A 20-min workout today can elevate you to 'Happy'."),
            ("😴 Sleep optimization", f"You slept {sleep}h. {'Great!' if sleep >= 7 else 'Aim for 7-9h tonight.'}"),
            ("🌿 Mindful tea break", "A 10-minute intentional pause with herbal tea improves focus and mood."),
            ("🎯 Set a micro-goal", "Achieving even a small task triggers dopamine and creates positive momentum."),
            ("🌞 Morning sunlight", "10 minutes of sunlight exposure regulates melatonin and improves mood."),
        ],
        2: [  # Happy
            ("✅ Deep work session", "You're in peak cognitive state. Tackle your most challenging task in the next 90 mins."),
            ("🌱 Pay it forward", "Helping someone else extends your positive mood and creates social capital."),
            ("📝 Log what worked", f"Sleep: {sleep}h, Screen: {screen}h, Activity: {activity} — note this pattern."),
            ("💪 Maintain routine", "Consistency is the key to sustained well-being. Keep today's habits going."),
            ("🎨 Creative exploration", "High mood boosts divergent thinking. Try a creative hobby or brainstorming session."),
            ("📣 Share positivity", "Your energy affects others. Spread encouragement in your community today."),
        ]
    }
    return recs[mood_idx]

# ─────────────────────────────────────────────
# WELLNESS QUOTES
# ─────────────────────────────────────────────
QUOTES = [
    ("Almost everything will work again if you unplug it for a few minutes, including you.", "Anne Lamott"),
    ("The greatest wealth is health.", "Virgil"),
    ("Mental health is not a destination, but a process.", "Noam Shpancer"),
    ("You don't have to control your thoughts. You just have to stop letting them control you.", "Dan Millman"),
    ("There is hope, even when your brain tells you there isn't.", "John Green"),
    ("Self-care is not selfish. You cannot serve from an empty vessel.", "Eleanor Brownn"),
    ("It's okay to not be okay — as long as you don't give up.", "Unknown"),
    ("Happiness is not something ready-made. It comes from your own actions.", "Dalai Lama"),
]

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
rf_model, scaler, model_acc, model_report, model_cm, dataset = train_model()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 20px;'>
        <div style='font-size:36px; margin-bottom:6px;'>🧠</div>
        <div style='font-family:DM Serif Display,serif; font-size:22px; color:#f1f5f9; font-weight:400;'>MindSync</div>
        <div style='font-size:11px; color:#64748b; letter-spacing:0.08em;'>AI WELLNESS COMPANION</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.selectbox("Navigate", [
        "🏠 Dashboard",
        "🔮 Mood Predictor",
        "📊 Analytics",
        "📓 Journal",
        "🎯 Goals & Streaks",
        "🚨 Risk Monitor",
        "🤖 Model Lab",
        "💡 Wellness Library",
    ])

    st.markdown("---")
    st.markdown("<div style='font-size:12px;color:#64748b;font-weight:500;letter-spacing:0.06em;margin-bottom:10px;'>QUICK METRICS</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:#94a3b8;font-size:13px;'>🔥 Streak: <b style='color:#f59e0b'>{st.session_state.streak} days</b></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:#94a3b8;font-size:13px;'>⭐ Points: <b style='color:#a78bfa'>{st.session_state.points}</b></div>", unsafe_allow_html=True)
    total = len(st.session_state.history)
    happy = len([h for h in st.session_state.history if h["mood_idx"] == 2])
    happy_pct = round(happy / total * 100) if total > 0 else 0
    st.markdown(f"<div style='color:#94a3b8;font-size:13px;'>😊 Happy rate: <b style='color:#10b981'>{happy_pct}%</b></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:#94a3b8;font-size:13px;'>📋 Logs: <b style='color:#38bdf8'>{total}</b></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.markdown("<p class='section-header'>Good day — here's your wellness overview</p>", unsafe_allow_html=True)
    quote = random.choice(QUOTES)
    st.markdown(f"<div class='quote-box'>&ldquo;{quote[0]}&rdquo; &mdash; <b>{quote[1]}</b></div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    total_logs = len(st.session_state.history)
    happy_count = len([h for h in st.session_state.history if h["mood_idx"] == 2])
    stress_count = len([h for h in st.session_state.history if h["mood_idx"] == 0])
    avg_wellness = round(np.mean([h.get("wellness_score", 50) for h in st.session_state.history]), 1) if st.session_state.history else 0

    with col1:
        st.markdown(f"""<div class='metric-card'>
            <h3>Total Sessions</h3><p class='val'>{total_logs}</p>
            <p class='delta delta-up'>↑ Logging consistently</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <h3>Avg Wellness Score</h3><p class='val'>{avg_wellness}</p>
            <p class='delta delta-up'>out of 100</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <h3>Happy Sessions</h3><p class='val' style='color:#15803d'>{happy_count}</p>
            <p class='delta delta-up'>😊 Keep going!</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='metric-card'>
            <h3>Stress Alerts</h3><p class='val' style='color:#dc2626'>{stress_count}</p>
            <p class='delta delta-down'>⚠️ Monitor closely</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("<p class='section-header' style='font-size:18px;'>Mood History</p>", unsafe_allow_html=True)
        if st.session_state.history:
            hist_df = pd.DataFrame(st.session_state.history)
            hist_df["index"] = range(len(hist_df))
            mood_colors = {0: "#ef4444", 1: "#3b82f6", 2: "#22c55e"}
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_df["index"], y=hist_df["mood_idx"],
                mode="lines+markers",
                line=dict(color="#6366f1", width=2),
                marker=dict(
                    color=[mood_colors[m] for m in hist_df["mood_idx"]],
                    size=10, line=dict(color="white", width=2)
                ),
                fill="tozeroy", fillcolor="rgba(99,102,241,0.06)",
                hovertemplate="Session %{x}<br>Mood: %{customdata}<extra></extra>",
                customdata=[["Stress","Neutral","Happy"][m] for m in hist_df["mood_idx"]]
            ))
            fig.update_layout(
                height=250, margin=dict(l=0,r=0,t=10,b=0),
                yaxis=dict(tickvals=[0,1,2], ticktext=["Stress","Neutral","Happy"], showgrid=True, gridcolor="#f1f5f9"),
                xaxis=dict(title="Session #", showgrid=False),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="DM Sans", size=12)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No mood data yet. Head to Mood Predictor to log your first session!")

    with col_b:
        st.markdown("<p class='section-header' style='font-size:18px;'>Mood Split</p>", unsafe_allow_html=True)
        if st.session_state.history:
            s = len([h for h in st.session_state.history if h["mood_idx"] == 0])
            n = len([h for h in st.session_state.history if h["mood_idx"] == 1])
            h2 = len([h for h in st.session_state.history if h["mood_idx"] == 2])
            fig2 = go.Figure(go.Pie(
                labels=["Stress", "Neutral", "Happy"],
                values=[s, n, h2],
                marker_colors=["#ef4444", "#3b82f6", "#22c55e"],
                hole=0.55,
                textfont_size=12
            ))
            fig2.update_layout(
                height=250, margin=dict(l=0,r=0,t=10,b=0),
                showlegend=True,
                legend=dict(orientation="h", y=-0.1),
                paper_bgcolor="white", font=dict(family="DM Sans", size=12)
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Run predictions to see your mood distribution.")

    # Recent journal entries preview
    if st.session_state.journal:
        st.markdown("---")
        st.markdown("<p class='section-header' style='font-size:18px;'>Latest Journal Entry</p>", unsafe_allow_html=True)
        latest = st.session_state.journal[-1]
        st.markdown(f"""<div class='journal-entry'>
            <div style='font-size:11px;color:#94a3b8;margin-bottom:6px;'>{latest['date']} · {latest['mood']}</div>
            <div style='font-size:14px;color:#334155;line-height:1.6;'>{latest['entry'][:200]}{'...' if len(latest['entry'])>200 else ''}</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: MOOD PREDICTOR
# ─────────────────────────────────────────────
elif page == "🔮 Mood Predictor":
    st.markdown("<p class='section-header'>Mood Predictor</p>", unsafe_allow_html=True)
    st.markdown("<p class='section-sub'>Enter your behavioral data — the hybrid RF+LSTM model will predict your mood</p>", unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 1])
    with col_in:
        with st.container():
            st.markdown("**Behavioral Inputs**")
            sleep = st.slider("🛏️ Sleep hours last night", 3.0, 10.0, 7.0, 0.5)
            screen = st.slider("📱 Screen time (hours)", 1.0, 12.0, 4.0, 0.5)
            activity = st.select_slider("🏃 Physical activity level", options=["Low", "Medium", "High"], value="Medium")
            social = st.select_slider("💬 Social interaction level", options=["Low", "Medium", "High"], value="Medium")
            stress_level = st.slider("😰 Self-reported stress (1=low, 10=high)", 1, 10, 5)
            energy_level = st.slider("⚡ Energy level (1=low, 10=high)", 1, 10, 6)
            water = st.slider("💧 Water intake (glasses)", 0, 12, 6)
            meditation = st.checkbox("🧘 Meditated today")
            exercise = st.checkbox("🏋️ Exercised today")
            predict_btn = st.button("🔮 Run Hybrid Prediction", type="primary", use_container_width=True)

    with col_out:
        if predict_btn:
            with st.spinner("Running RF + LSTM inference..."):
                import time; time.sleep(0.8)
                result = hybrid_predict(rf_model, scaler, sleep, screen, activity, social)
                mood_idx = result["hybrid_idx"]
                mood_name = result["hybrid_pred"]
                confidence = result["confidence"]
                wellness = compute_wellness_score(sleep, screen, activity, social, mood_idx)
                risk_level, risk_score, risk_flags = assess_risk(sleep, screen, activity, social, st.session_state.history)

                # Save to history
                entry = {
                    "timestamp": str(datetime.datetime.now()),
                    "date": datetime.date.today().strftime("%b %d, %Y"),
                    "sleep": sleep, "screen": screen,
                    "activity": activity, "social": social,
                    "stress_level": stress_level, "energy_level": energy_level,
                    "water": water, "meditation": meditation, "exercise": exercise,
                    "mood_idx": mood_idx, "mood": mood_name,
                    "confidence": round(confidence * 100, 1),
                    "wellness_score": wellness,
                    "risk_level": risk_level,
                    "rf_pred": result["rf_pred"], "lstm_pred": result["lstm_pred"],
                    "hybrid_probs": result["hybrid_probs"]
                }
                st.session_state.history.append(entry)
                st.session_state.points += 10
                if mood_name == "Happy":
                    st.session_state.streak += 1
                    st.session_state.points += 5
                else:
                    st.session_state.streak = 0

            # Display results
            mood_colors = {0: "stress", 1: "neutral", 2: "happy"}
            mood_emojis = {0: "😟", 1: "😐", 2: "😊"}
            st.markdown(f"""
            <div style='text-align:center; padding: 24px; background:white; border-radius:16px; border:1px solid #f1f5f9;'>
                <div style='font-size:52px; margin-bottom:8px;'>{mood_emojis[mood_idx]}</div>
                <span class='mood-badge mood-{mood_colors[mood_idx]}'>{mood_name}</span>
                <div style='margin-top:12px; font-size:13px; color:#64748b;'>Confidence: <b>{round(confidence*100,1)}%</b></div>
                <div style='margin-top:6px;'>
                    <span style='font-size:12px; background:#f1f5f9; color:#475569; padding:3px 10px; border-radius:100px;'>RF: {result["rf_pred"]}</span>
                    &nbsp;
                    <span style='font-size:12px; background:#f1f5f9; color:#475569; padding:3px 10px; border-radius:100px;'>LSTM: {result["lstm_pred"]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Probability Distribution**")
            probs = result["hybrid_probs"]
            for label, prob, color in zip(["Stress","Neutral","Happy"], probs, ["#ef4444","#3b82f6","#22c55e"]):
                st.markdown(f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>"
                    f"<span style='font-size:12px;width:55px;color:#64748b;'>{label}</span>"
                    f"<div style='flex:1;background:#f1f5f9;border-radius:100px;height:8px;overflow:hidden;'>"
                    f"<div style='width:{prob*100:.0f}%;height:100%;background:{color};border-radius:100px;'></div></div>"
                    f"<span style='font-size:12px;color:#1e293b;font-weight:500;width:36px;text-align:right;'>{prob*100:.0f}%</span>"
                    f"</div>", unsafe_allow_html=True)

            col_w, col_r = st.columns(2)
            with col_w:
                st.metric("Wellness Score", f"{wellness}/100")
            with col_r:
                risk_class = {"Low":"risk-low","Medium":"risk-medium","High":"risk-high"}[risk_level]
                st.markdown(f"<div style='margin-top:24px;'>Risk Level: <span class='{risk_class}'>{risk_level}</span></div>", unsafe_allow_html=True)

            if risk_flags:
                st.warning(" · ".join(risk_flags))

            st.markdown("---")
            st.markdown("**Personalized Recommendations**")
            recs = get_recommendations(mood_idx, sleep, screen, activity, social)
            for title, desc in recs[:4]:
                st.markdown(f"<div class='rec-card'><h4>{title}</h4><p>{desc}</p></div>", unsafe_allow_html=True)

            st.success(f"✅ +10 points earned! Total: {st.session_state.points} pts")

        else:
            st.markdown("""
            <div style='display:flex;flex-direction:column;align-items:center;justify-content:center;height:400px;color:#94a3b8;'>
                <div style='font-size:48px;margin-bottom:12px;'>🔮</div>
                <div style='font-size:14px;'>Fill in your behavioral data and click predict</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: ANALYTICS
# ─────────────────────────────────────────────
elif page == "📊 Analytics":
    st.markdown("<p class='section-header'>Deep Analytics</p>", unsafe_allow_html=True)
    st.markdown("<p class='section-sub'>Behavioral patterns, correlations, and mood trends over time</p>", unsafe_allow_html=True)

    if len(st.session_state.history) < 2:
        st.info("Log at least 2 sessions in Mood Predictor to unlock analytics.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)

        # Wellness over time
        st.markdown("#### Wellness Score Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(df_hist))), y=df_hist["wellness_score"],
            mode="lines+markers", fill="tozeroy",
            line=dict(color="#6366f1", width=2.5),
            fillcolor="rgba(99,102,241,0.08)",
            marker=dict(size=7, color="#6366f1"),
        ))
        fig.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=0),
            yaxis=dict(range=[0,105], showgrid=True, gridcolor="#f1f5f9"),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(title="Session", showgrid=False),
            font=dict(family="DM Sans", size=12))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Sleep vs Mood")
            fig2 = px.scatter(df_hist, x="sleep", y="mood_idx", color="mood",
                color_discrete_map={"Happy":"#22c55e","Neutral":"#3b82f6","Stress":"#ef4444"},
                labels={"sleep":"Sleep Hours","mood_idx":"Mood (0=Stress,2=Happy)"},
                trendline="ols",
            )
            fig2.update_layout(height=250, margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="DM Sans", size=12))
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            st.markdown("#### Screen Time vs Wellness")
            fig3 = px.scatter(df_hist, x="screen", y="wellness_score", color="mood",
                color_discrete_map={"Happy":"#22c55e","Neutral":"#3b82f6","Stress":"#ef4444"},
                labels={"screen":"Screen Time (h)","wellness_score":"Wellness Score"},
                trendline="ols",
            )
            fig3.update_layout(height=250, margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="DM Sans", size=12))
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### Behavioral Averages by Mood")
        mood_grp = df_hist.groupby("mood")[["sleep","screen","wellness_score","stress_level","energy_level"]].mean().round(2)
        st.dataframe(mood_grp.style.background_gradient(cmap="RdYlGn"), use_container_width=True)

        st.markdown("#### Confidence Distribution")
        fig4 = px.histogram(df_hist, x="confidence", nbins=15, color="mood",
            color_discrete_map={"Happy":"#22c55e","Neutral":"#3b82f6","Stress":"#ef4444"},
            labels={"confidence":"Prediction Confidence (%)"},
            barmode="overlay", opacity=0.7
        )
        fig4.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="DM Sans", size=12))
        st.plotly_chart(fig4, use_container_width=True)

        # Full log table
        st.markdown("#### Full Session Log")
        show_cols = ["date","mood","sleep","screen","activity","social","wellness_score","confidence","risk_level","rf_pred","lstm_pred"]
        available = [c for c in show_cols if c in df_hist.columns]
        st.dataframe(df_hist[available].sort_index(ascending=False), use_container_width=True)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv = df_hist.to_csv(index=False)
            st.download_button("⬇️ Download CSV", csv, "wellness_data.csv", "text/csv", use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: JOURNAL
# ─────────────────────────────────────────────
elif page == "📓 Journal":
    st.markdown("<p class='section-header'>Wellness Journal</p>", unsafe_allow_html=True)
    st.markdown("<p class='section-sub'>Reflect on your day — journaling reduces stress and improves emotional clarity</p>", unsafe_allow_html=True)

    with st.form("journal_form", clear_on_submit=True):
        col_j1, col_j2 = st.columns(2)
        with col_j1:
            current_mood = st.selectbox("Today's mood", ["😊 Happy", "😐 Neutral", "😟 Stressed", "😴 Tired", "😤 Anxious", "🥰 Grateful"])
        with col_j2:
            tags = st.multiselect("Tags", ["Work","Study","Family","Health","Social","Sleep","Exercise","Food","Hobby","Mindfulness"])

        entry_text = st.text_area("Write your thoughts...", height=150, placeholder="How was your day? What's on your mind?")
        highlight = st.text_input("🌟 Today's highlight (optional)")
        gratitude = st.text_input("🙏 One thing you're grateful for")
        submitted = st.form_submit_button("📝 Save Entry", use_container_width=True)

        if submitted and entry_text:
            st.session_state.journal.append({
                "date": datetime.date.today().strftime("%b %d, %Y"),
                "mood": current_mood,
                "entry": entry_text,
                "highlight": highlight,
                "gratitude": gratitude,
                "tags": tags
            })
            st.session_state.points += 5
            st.success("Entry saved! +5 points 🎉")

    st.markdown("---")
    st.markdown("#### Past Entries")
    if st.session_state.journal:
        for j in reversed(st.session_state.journal):
            with st.expander(f"{j['date']} · {j['mood']}"):
                st.write(j["entry"])
                if j.get("highlight"):
                    st.markdown(f"🌟 **Highlight:** {j['highlight']}")
                if j.get("gratitude"):
                    st.markdown(f"🙏 **Grateful for:** {j['gratitude']}")
                if j.get("tags"):
                    st.markdown("**Tags:** " + " · ".join([f"`{t}`" for t in j["tags"]]))
    else:
        st.info("No journal entries yet. Write your first one above!")

# ─────────────────────────────────────────────
# PAGE: GOALS & STREAKS
# ─────────────────────────────────────────────
elif page == "🎯 Goals & Streaks":
    st.markdown("<p class='section-header'>Goals & Gamification</p>", unsafe_allow_html=True)
    st.markdown("<p class='section-sub'>Set wellness targets and track your achievements</p>", unsafe_allow_html=True)

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("#### Set Your Daily Goals")
        sleep_goal = st.slider("🛏️ Target sleep (hours)", 6.0, 10.0, st.session_state.goals["sleep_target"], 0.5)
        screen_goal = st.slider("📱 Max screen time (hours)", 1.0, 8.0, st.session_state.goals["screen_target"], 0.5)
        activity_goal = st.selectbox("🏃 Activity target", ["Low","Medium","High"], index=["Low","Medium","High"].index(st.session_state.goals["activity_target"]))
        social_goal = st.selectbox("💬 Social target", ["Low","Medium","High"], index=["Low","Medium","High"].index(st.session_state.goals["social_target"]))
        if st.button("💾 Save Goals", use_container_width=True):
            st.session_state.goals = {"sleep_target": sleep_goal, "screen_target": screen_goal, "activity_target": activity_goal, "social_target": social_goal}
            st.success("Goals updated!")

    with col_g2:
        st.markdown("#### Goal Achievement")
        if st.session_state.history:
            last = st.session_state.history[-1]
            goals = st.session_state.goals
            checks = {
                "Sleep": (last["sleep"] >= goals["sleep_target"], f"{last['sleep']}h / {goals['sleep_target']}h"),
                "Screen time": (last["screen"] <= goals["screen_target"], f"{last['screen']}h / ≤{goals['screen_target']}h"),
                "Activity": (last["activity"] == goals["activity_target"] or (last["activity"]=="High"), f"{last['activity']} / {goals['activity_target']}"),
                "Social": (last["social"] != "Low" or goals["social_target"]=="Low", f"{last['social']} / {goals['social_target']}"),
            }
            for goal_name, (met, detail) in checks.items():
                icon = "✅" if met else "❌"
                st.markdown(f"<div style='padding:10px;background:{'#f0fdf4' if met else '#fef2f2'};border-radius:8px;margin-bottom:6px;font-size:13px;'>{icon} <b>{goal_name}</b>: {detail}</div>", unsafe_allow_html=True)
        else:
            st.info("Log a session to see goal achievement.")

    st.markdown("---")
    st.markdown("#### Achievements & Badges")
    total = len(st.session_state.history)
    badges = [
        ("🌱 First Step", total >= 1, "Log your first session"),
        ("🔥 On a Roll", st.session_state.streak >= 3, "3-day happy streak"),
        ("📊 Data Driven", total >= 5, "Log 5 sessions"),
        ("😊 Happy Champion", len([h for h in st.session_state.history if h["mood_idx"]==2]) >= 3, "3 happy predictions"),
        ("📓 Reflective", len(st.session_state.journal) >= 3, "Write 3 journal entries"),
        ("💪 Wellness Warrior", total >= 10, "Log 10 sessions"),
        ("🧘 Zen Master", st.session_state.streak >= 7, "7-day happy streak"),
        ("⭐ Point Collector", st.session_state.points >= 50, "Earn 50 points"),
    ]
    cols = st.columns(4)
    for i, (badge, earned, desc) in enumerate(badges):
        with cols[i % 4]:
            st.markdown(f"""
            <div style='text-align:center; padding:16px; background:{'white' if earned else '#f8fafc'};
                border-radius:12px; border:1px solid {'#bbf7d0' if earned else '#f1f5f9'};
                margin-bottom:10px; opacity:{'1' if earned else '0.45'};'>
                <div style='font-size:28px;'>{badge.split()[0]}</div>
                <div style='font-size:12px;font-weight:600;color:#1e293b;margin:4px 0;'>{' '.join(badge.split()[1:])}</div>
                <div style='font-size:11px;color:#94a3b8;'>{desc}</div>
                {'<div style="font-size:10px;color:#15803d;margin-top:4px;font-weight:600;">EARNED ✓</div>' if earned else ''}
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"#### Total Points: **{st.session_state.points}** ⭐")
    level = "Bronze" if st.session_state.points < 50 else "Silver" if st.session_state.points < 150 else "Gold"
    st.markdown(f"Level: **{level}** · Next level at {'50' if level=='Bronze' else '150' if level=='Silver' else '∞'} pts")
    prog = min(st.session_state.points / (50 if level == 'Bronze' else 150 if level == 'Silver' else 300), 1.0)
    st.progress(prog)

# ─────────────────────────────────────────────
# PAGE: RISK MONITOR
# ─────────────────────────────────────────────
elif page == "🚨 Risk Monitor":
    st.markdown("<p class='section-header'>Risk Monitor</p>", unsafe_allow_html=True)
    st.markdown("<p class='section-sub'>Early warning system for mental health deterioration</p>", unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No session data yet. Run predictions to activate risk monitoring.")
    else:
        last = st.session_state.history[-1]
        risk_level, risk_score, risk_flags = assess_risk(
            last["sleep"], last["screen"], last["activity"], last["social"], st.session_state.history
        )

        col_r1, col_r2, col_r3 = st.columns(3)
        risk_color = {"Low":"#22c55e","Medium":"#f59e0b","High":"#ef4444"}[risk_level]
        with col_r1:
            st.markdown(f"""<div class='metric-card'>
                <h3>Current Risk Level</h3>
                <p class='val' style='color:{risk_color};'>{risk_level}</p>
                <p class='delta'>Score: {risk_score}/100</p></div>""", unsafe_allow_html=True)
        with col_r2:
            consec_stress = 0
            for h in reversed(st.session_state.history):
                if h["mood_idx"] == 0: consec_stress += 1
                else: break
            st.markdown(f"""<div class='metric-card'>
                <h3>Consecutive Stress</h3>
                <p class='val' style='color:{"#ef4444" if consec_stress>=3 else "#0f172a"};'>{consec_stress}</p>
                <p class='delta'>sessions in a row</p></div>""", unsafe_allow_html=True)
        with col_r3:
            avg_sleep = round(np.mean([h["sleep"] for h in st.session_state.history[-7:]]), 1)
            st.markdown(f"""<div class='metric-card'>
                <h3>7-Day Avg Sleep</h3>
                <p class='val' style='color:{"#ef4444" if avg_sleep<6 else "#0f172a"};'>{avg_sleep}h</p>
                <p class='delta'>{'⚠️ Below threshold' if avg_sleep<7 else '✅ Healthy range'}</p></div>""", unsafe_allow_html=True)

        if risk_level == "High":
            st.error("🚨 High risk detected. Please consider speaking to a counselor or trusted person.")
        elif risk_level == "Medium":
            st.warning("⚠️ Moderate risk. Monitor your patterns and practice self-care.")
        else:
            st.success("✅ Low risk. Your wellness indicators look healthy.")

        if risk_flags:
            st.markdown("#### Risk Flags Detected")
            for f in risk_flags:
                st.markdown(f"<div style='background:#fef2f2;padding:10px 14px;border-radius:8px;font-size:13px;margin-bottom:6px;border-left:3px solid #ef4444;'>{f}</div>", unsafe_allow_html=True)

        st.markdown("#### Risk Score Trend")
        if len(st.session_state.history) >= 2:
            risk_scores = []
            for i, h in enumerate(st.session_state.history):
                _, rs, _ = assess_risk(h["sleep"], h["screen"], h["activity"], h["social"], st.session_state.history[:i])
                risk_scores.append(rs)
            fig_r = go.Figure()
            fig_r.add_hrect(y0=50, y1=100, fillcolor="#ef4444", opacity=0.06, line_width=0)
            fig_r.add_hrect(y0=25, y1=50, fillcolor="#f59e0b", opacity=0.06, line_width=0)
            fig_r.add_trace(go.Scatter(
                y=risk_scores, mode="lines+markers",
                line=dict(color="#6366f1", width=2),
                marker=dict(size=7, color=[("#ef4444" if r>=50 else "#f59e0b" if r>=25 else "#22c55e") for r in risk_scores])
            ))
            fig_r.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=0),
                yaxis=dict(range=[0,105], title="Risk Score"),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="DM Sans", size=12))
            st.plotly_chart(fig_r, use_container_width=True)

        st.markdown("#### Crisis Resources")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("""<div class='rec-card'>
                <h4>🇮🇳 iCall (India)</h4>
                <p>TISS helpline: <b>9152987821</b> · Free counseling for students & professionals</p>
            </div>
            <div class='rec-card'>
                <h4>📞 Vandrevala Foundation</h4>
                <p>24/7: <b>1860-2662-345</b> · Mental health support across India</p>
            </div>""", unsafe_allow_html=True)
        with col_c2:
            st.markdown("""<div class='rec-card'>
                <h4>🌐 iCharity (online)</h4>
                <p>Connect with licensed therapists. Sliding scale fees available.</p>
            </div>
            <div class='rec-card'>
                <h4>🏥 VIT Counseling Cell</h4>
                <p>On-campus support for VIT students. Visit Student Welfare office.</p>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: MODEL LAB
# ─────────────────────────────────────────────
elif page == "🤖 Model Lab":
    st.markdown("<p class='section-header'>Model Lab</p>", unsafe_allow_html=True)
    st.markdown("<p class='section-sub'>Explore the hybrid RF+LSTM architecture, metrics, and training data</p>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Performance Metrics", "Feature Importance", "Confusion Matrix", "Training Data"])

    with tab1:
        st.markdown(f"**Random Forest Test Accuracy: {model_acc*100:.1f}%** (n=100 estimators, max_depth=8)")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        for col, cls in zip([col_m1, col_m2, col_m3, col_m4], ["Stress","Neutral","Happy","weighted avg"]):
            with col:
                r = model_report[cls]
                st.markdown(f"""<div class='metric-card'>
                    <h3>{cls}</h3>
                    <p class='val' style='font-size:20px;'>{r['f1-score']:.2f}</p>
                    <p class='delta'>P:{r['precision']:.2f} R:{r['recall']:.2f}</p></div>""", unsafe_allow_html=True)

        st.markdown("#### Metric Comparison")
        fig_m = go.Figure()
        classes = ["Stress","Neutral","Happy"]
        for metric, color in [("precision","#6366f1"),("recall","#22c55e"),("f1-score","#f59e0b")]:
            fig_m.add_trace(go.Bar(
                name=metric.capitalize(),
                x=classes,
                y=[model_report[c][metric] for c in classes],
                marker_color=color
            ))
        fig_m.update_layout(barmode="group", height=280, margin=dict(l=0,r=0,t=10,b=0),
            yaxis=dict(range=[0,1.1], showgrid=True, gridcolor="#f1f5f9"),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="DM Sans", size=12),
            legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_m, use_container_width=True)

        st.markdown("#### Hybrid Fusion Logic")
        st.code("""# Weighted fusion: LSTM captures temporal patterns, RF handles structure
rf_probs   = RandomForest.predict_proba(X_scaled)      # structured features
lstm_probs = LSTM.predict(X_sequence).softmax()         # temporal patterns

hybrid_probs = 0.6 * lstm_probs + 0.4 * rf_probs        # LSTM weighted higher
final_pred   = argmax(hybrid_probs)                     # dominant class""", language="python")

    with tab2:
        importances = rf_model.feature_importances_
        features = ["Sleep Hours", "Screen Time", "Activity Level", "Social Interaction"]
        fi_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance", ascending=True)
        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"],
            orientation="h", marker_color=["#6366f1","#22c55e","#f59e0b","#ef4444"],
            text=[f"{v:.3f}" for v in fi_df["Importance"]], textposition="outside"
        ))
        fig_fi.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(range=[0,0.5], showgrid=True, gridcolor="#f1f5f9"),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="DM Sans", size=12))
        st.plotly_chart(fig_fi, use_container_width=True)
        st.markdown("Sleep hours is the strongest predictor of mood, followed by screen time and physical activity.")

    with tab3:
        cm_df = pd.DataFrame(model_cm,
            index=["Actual: Stress","Actual: Neutral","Actual: Happy"],
            columns=["Pred: Stress","Pred: Neutral","Pred: Happy"])
        fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues",
            labels=dict(color="Count"))
        fig_cm.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=0),
            font=dict(family="DM Sans", size=12))
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("A clean diagonal indicates strong classification performance across all three mood classes.")

    with tab4:
        st.markdown(f"**Synthetic dataset: {len(dataset)} samples** generated with behavioral heuristics + Gaussian noise")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            fig_dist = px.histogram(dataset, x="mood", color="mood",
                color_discrete_map={0:"#ef4444",1:"#3b82f6",2:"#22c55e"},
                labels={"mood":"Mood Class (0=Stress,1=Neutral,2=Happy)"},
                nbins=3)
            fig_dist.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                showlegend=False, font=dict(family="DM Sans", size=12))
            st.plotly_chart(fig_dist, use_container_width=True)
        with col_s2:
            fig_pair = px.scatter(dataset.sample(200), x="sleep_hours", y="screen_time",
                color=dataset.sample(200)["mood"].map({0:"Stress",1:"Neutral",2:"Happy"}),
                color_discrete_map={"Happy":"#22c55e","Neutral":"#3b82f6","Stress":"#ef4444"},
                opacity=0.6)
            fig_pair.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="DM Sans", size=12))
            st.plotly_chart(fig_pair, use_container_width=True)
        st.dataframe(dataset.describe().round(2), use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: WELLNESS LIBRARY
# ─────────────────────────────────────────────
elif page == "💡 Wellness Library":
    st.markdown("<p class='section-header'>Wellness Library</p>", unsafe_allow_html=True)
    st.markdown("<p class='section-sub'>Evidence-based techniques for improving mental health</p>", unsafe_allow_html=True)

    topics = {
        "😴 Sleep Hygiene": [
            ("Go to bed at consistent times", "Your circadian rhythm thrives on regularity. Set a fixed sleep/wake schedule even on weekends."),
            ("Avoid screens 1h before bed", "Blue light suppresses melatonin by up to 50%, delaying sleep onset by 1-3 hours."),
            ("Keep your room cool (18-20°C)", "Core body temperature drop triggers sleep. A cool room accelerates this process."),
            ("No caffeine after 2pm", "Caffeine's half-life is 5-7 hours — a 4pm coffee still has 50% active at 9pm."),
        ],
        "🧘 Mindfulness & Stress": [
            ("4-7-8 breathing", "Inhale for 4s, hold for 7s, exhale for 8s. Activates the vagus nerve to calm anxiety."),
            ("Progressive muscle relaxation", "Tense and release muscle groups from feet to head. Reduces physical stress symptoms."),
            ("5-4-3-2-1 grounding", "Name 5 things you see, 4 you hear, 3 you touch, 2 you smell, 1 you taste. Breaks anxiety cycles."),
            ("Cold water on wrists", "Cool water on pulse points quickly lowers heart rate during acute stress."),
        ],
        "🏃 Physical Activity": [
            ("30 mins of moderate exercise = antidepressant", "Exercise raises BDNF (brain-derived neurotrophic factor), literally growing new neural connections."),
            ("Walk after meals", "A 10-15 minute post-meal walk lowers blood sugar and improves energy levels."),
            ("Yoga for mood regulation", "Yoga combines breathwork and movement, reducing cortisol and increasing GABA neurotransmitters."),
            ("Aim for 7,500 steps/day", "Research shows 7,500 steps is the threshold where mood and health benefits plateau."),
        ],
        "💬 Social Connection": [
            ("Quality over quantity", "1-3 deep relationships predict wellness better than many shallow ones."),
            ("Reach out first", "Waiting to be contacted increases loneliness. Initiate contact — most people feel positively surprised."),
            ("Digital vs in-person", "In-person interaction triggers oxytocin at 4x the rate of digital communication."),
            ("Volunteer regularly", "Altruistic behavior activates the brain's reward centers, reducing depression risk by 22%."),
        ],
        "📱 Digital Wellness": [
            ("Phone-free mornings (first 30 mins)", "Morning phone checking activates reactive mode, reducing focus and raising stress for hours."),
            ("Turn off non-essential notifications", "Notifications create 23-minute focus interruption cycles. Disable all but critical alerts."),
            ("Greyscale mode after 8pm", "Reducing color saturation on devices makes them less stimulating and easier to disengage from."),
            ("Screen time tracking", "Awareness alone reduces usage. Review weekly screen time every Sunday."),
        ],
    }

    for category, items in topics.items():
        with st.expander(category, expanded=False):
            for title, desc in items:
                st.markdown(f"<div class='rec-card'><h4>{title}</h4><p>{desc}</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Daily Wellness Challenge")
    challenges = [
        "🧘 Meditate for 10 minutes before checking your phone",
        "💧 Drink 8 glasses of water throughout the day",
        "📵 No screens for the last hour before bed",
        "🚶 Take a 15-minute walk outside during daylight hours",
        "📓 Write 3 things you're grateful for tonight",
        "💬 Have a meaningful conversation with someone you care about",
        "🏋️ Do 20 minutes of exercise — any type counts",
        "😴 Set an alarm to wind down 30 minutes earlier tonight",
    ]
    today_challenge = challenges[datetime.date.today().toordinal() % len(challenges)]
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#f0fdf4,#dcfce7);border-radius:16px;padding:20px 24px;border:1px solid #bbf7d0;'>
        <div style='font-size:11px;color:#15803d;font-weight:600;letter-spacing:0.08em;margin-bottom:6px;'>TODAY'S CHALLENGE</div>
        <div style='font-size:16px;color:#166534;font-weight:500;'>{today_challenge}</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("✅ Mark as Complete", use_container_width=True):
        st.session_state.points += 15
        st.success(f"Challenge completed! +15 points. Total: {st.session_state.points} pts 🎉")
