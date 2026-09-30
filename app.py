import time
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + THEME
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XIDS — Live Intrusion Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
    .stApp { background-color: #0D1B2A; color: #E8EEF2; }
    section[data-testid="stSidebar"] { background-color: #16283D; }
    h1, h2, h3, h4 { color: #E8EEF2 !important; font-family: 'Consolas', monospace; }
    div[data-testid="stMetricValue"] { color: #1B9AAA; font-family: 'Consolas', monospace; }
    div[data-testid="stMetricLabel"] { color: #9DB0BD; }
    .status-active { color: #2ECC71; font-weight: bold; font-family: 'Consolas', monospace; }
    .alert-row-attack { background-color: rgba(240,80,60,0.15); padding: 6px; border-radius: 4px; }
    .alert-row-benign { background-color: rgba(27,154,170,0.08); padding: 6px; border-radius: 4px; }
    .ai-note {
        background-color: #1E3247; border-left: 4px solid #F0A202;
        padding: 14px; border-radius: 6px; font-family: 'Consolas', monospace;
        font-size: 14px; line-height: 1.5; margin-top: 10px;
    }
    .block-container { padding-top: 1.5rem; }
    hr { border-color: #29405A; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

GROQ_MODEL = "openai/gpt-oss-20b"   # swap to "openai/gpt-oss-120b" for richer (slower) explanations

# ─────────────────────────────────────────────────────────────────────────
# LOAD MODEL ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("model_artifacts/xgboost_xids_model_pso.pkl")
    scaler = joblib.load("model_artifacts/scaler.pkl")
    le = joblib.load("model_artifacts/label_encoder.pkl")
    selected_features = joblib.load("model_artifacts/pso_selected_features.pkl")
    selected_mask = joblib.load("model_artifacts/pso_selected_mask.pkl")
    all_feature_columns = joblib.load("model_artifacts/all_feature_columns.pkl")
    explainer = shap.TreeExplainer(model)
    return model, scaler, le, selected_features, selected_mask, all_feature_columns, explainer


@st.cache_data
def load_sample_flows():
    df = pd.read_csv("model_artifacts/sample_flows.csv")
    missing = [c for c in all_feature_columns if c not in df.columns]
    for col in missing:
        df[col] = 0.0
    return df[all_feature_columns]


try:
    model, scaler, le, selected_features, selected_mask, all_feature_columns, explainer = load_artifacts()
    sample_flows = load_sample_flows()
    ARTIFACTS_OK = True
except Exception as e:
    ARTIFACTS_OK = False
    LOAD_ERROR = str(e)

# ─────────────────────────────────────────────────────────────────────────
# GROQ CLIENT
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_groq_client():
    try:
        from groq import Groq
        api_key = st.secrets["GROQ_API_KEY"]
        return Groq(api_key=api_key)
    except Exception:
        return None


groq_client = get_groq_client()


def generate_ai_explanation(predicted_class, confidence, top_features):
    """Called ONLY for attack predictions. top_features: list of (name, shap_value)."""
    if groq_client is None:
        return "⚠ AI explanation unavailable — GROQ_API_KEY not configured in st.secrets."

    feature_lines = "\n".join(
        f"- {name}: SHAP impact {val:+.4f}" for name, val in top_features
    )
    prompt = f"""You are a network security analyst assistant. A network intrusion detection
system just flagged a traffic flow as an attack. Explain WHY in 2-3 short sentences,
in plain English, for a security analyst reading a dashboard alert. Be specific and
technical but concise. Do not repeat the raw numbers verbatim — interpret them.

Predicted attack type: {predicted_class}
Model confidence: {confidence:.1f}%

Top contributing features (SHAP explainability values, positive = pushed toward this
attack classification):
{feature_lines}

Write only the explanation, no preamble, no headers."""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=180,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠ AI explanation failed: {e}"


# ─────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────
defaults = {
    "flows_analyzed": 0,
    "alerts_triggered": 0,
    "latencies": [],
    "alert_log": [],
    "incident_log": [],   # full attack records -> exportable CSV
    "last_result": None,
    "live_mode": False,
    "start_time": time.time(),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def run_inference(raw_row_df):
    raw_row_df = raw_row_df.reindex(columns=all_feature_columns, fill_value=0.0)
    row_scaled = scaler.transform(raw_row_df[all_feature_columns])
    row_selected = row_scaled[:, selected_mask]

    t0 = time.time()
    pred = model.predict(row_selected)
    pred_proba = model.predict_proba(row_selected)
    t1 = time.time()
    shap_values = explainer.shap_values(row_selected)
    t2 = time.time()

    predicted_class = le.inverse_transform(pred)[0]
    confidence = float(pred_proba[0][pred[0]] * 100)

    return {
        "class": predicted_class,
        "confidence": confidence,
        "inference_ms": (t1 - t0) * 1000,
        "explain_ms": (t2 - t1) * 1000,
        "total_ms": (t2 - t0) * 1000,
        "shap_values": shap_values,
        "row_selected": row_selected,
    }


def get_top_shap_features(result, top_k=5):
    sv = result["shap_values"]
    if isinstance(sv, list):
        pred_class_idx = list(le.classes_).index(result["class"])
        vals = np.array(sv[pred_class_idx])[0]
    else:
        vals = np.array(sv)[0]
    order = np.argsort(np.abs(vals))[-top_k:][::-1]
    return [(selected_features[i], float(vals[i])) for i in order], vals, order


def log_result(result, source_label="live"):
    st.session_state.flows_analyzed += 1
    st.session_state.latencies.append(result["total_ms"])
    is_attack = result["class"].strip().lower() != "benign"
    ai_message = None

    if is_attack:
        st.session_state.alerts_triggered += 1
        top_feats, _, _ = get_top_shap_features(result, top_k=5)
        with st.spinner("🤖 Generating AI explanation..."):
            ai_message = generate_ai_explanation(result["class"], result["confidence"], top_feats)

        st.session_state.incident_log.insert(0, {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_class": result["class"],
            "confidence_pct": round(result["confidence"], 2),
            "total_latency_ms": round(result["total_ms"], 3),
            "inference_ms": round(result["inference_ms"], 3),
            "explanation_ms": round(result["explain_ms"], 3),
            "top_features": "; ".join(f"{n} ({v:+.4f})" for n, v in top_feats),
            "ai_explanation": ai_message,
            "source": source_label,
        })

    st.session_state.alert_log.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "class": result["class"],
        "confidence": f'{result["confidence"]:.1f}%',
        "latency_ms": f'{result["total_ms"]:.2f}',
        "is_attack": is_attack,
        "source": source_label,
    })
    st.session_state.alert_log = st.session_state.alert_log[:25]
    result["ai_message"] = ai_message
    result["is_attack"] = is_attack
    st.session_state.last_result = result


# ─────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────
uptime = int(time.time() - st.session_state.start_time)
h, rem = divmod(uptime, 3600)
m, s = divmod(rem, 60)

header_l, header_r = st.columns([3, 1])
with header_l:
    st.markdown("## 🛡️ XIDS — Real-Time Network Intrusion Monitoring")
    st.caption("XGBoost + PSO Feature Selection + SHAP · AI-generated analyst explanations via Groq")
with header_r:
    st.markdown(
        f"<div style='text-align:right'>"
        f"<span class='status-active'>● SYSTEM ACTIVE</span><br>"
        f"<span style='color:#9DB0BD;font-family:monospace'>Uptime {h:02d}:{m:02d}:{s:02d}</span>"
        f"</div>", unsafe_allow_html=True
    )
st.markdown("<hr>", unsafe_allow_html=True)

if not ARTIFACTS_OK:
    st.error(f"Model artifacts not found. Details: {LOAD_ERROR}")
    st.stop()

if groq_client is None:
    st.warning(
        "⚠ GROQ_API_KEY not found in st.secrets — AI explanations will show a placeholder "
        "message until you add it (Streamlit Cloud: App settings → Secrets)."
    )

# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    mode = st.radio("Mode", ["🔴 Live Traffic Simulation", "🧪 Manual Flow Testing"])
    st.markdown("---")

    if mode.startswith("🔴"):
        refresh_rate = st.slider("Simulated flow interval (seconds)", 1.0, 5.0, 2.0, 0.5)
        st.session_state.live_mode = st.toggle("▶ Start live monitoring", value=st.session_state.live_mode)
    else:
        st.session_state.live_mode = False

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.caption(f"**Classifier:** XGBoost  |  **Explainer:** SHAP TreeExplainer")
    st.caption(f"**Features:** {len(selected_features)} (PSO-selected)  |  **Classes:** {len(le.classes_)}")
    st.caption(f"**AI model:** `{GROQ_MODEL}` (Groq)")

    st.markdown("---")
    st.markdown("### 📥 Incident Report")
    n_incidents = len(st.session_state.incident_log)
    st.caption(f"{n_incidents} attack incident(s) logged this session")
    if n_incidents > 0:
        incident_df = pd.DataFrame(st.session_state.incident_log)
        st.download_button(
            "⬇️ Download Incident Report (CSV)",
            data=incident_df.to_csv(index=False).encode("utf-8"),
            file_name=f"xids_incident_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if st.button("🔄 Reset session stats", use_container_width=True):
        for k in defaults:
            st.session_state[k] = defaults[k] if not isinstance(defaults[k], list) else []
        st.session_state.start_time = time.time()
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────
# TOP METRICS
# ─────────────────────────────────────────────────────────────────────────
avg_latency = np.mean(st.session_state.latencies) if st.session_state.latencies else 0
detection_rate = (
    (st.session_state.alerts_triggered / st.session_state.flows_analyzed * 100)
    if st.session_state.flows_analyzed else 0
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Flows Analyzed", f"{st.session_state.flows_analyzed:,}")
m2.metric("Alerts Triggered", f"{st.session_state.alerts_triggered:,}")
m3.metric("Avg Latency", f"{avg_latency:.2f} ms")
m4.metric("Alert Rate", f"{detection_rate:.1f}%")
st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────
left, right = st.columns([1.3, 1])

with left:
    if mode.startswith("🔴"):
        st.markdown("#### 📡 Incoming Flow")
        placeholder_flow = st.empty()

        if st.session_state.live_mode:
            row = sample_flows.sample(1).reset_index(drop=True)
            result = run_inference(row)
            log_result(result, source_label="live")

        if st.session_state.last_result:
            r = st.session_state.last_result
            badge_color = "#F0503C" if r["is_attack"] else "#2ECC71"
            badge_text = "⚠ THREAT DETECTED" if r["is_attack"] else "✓ NORMAL TRAFFIC"

            with placeholder_flow.container():
                st.markdown(
                    f"<div style='padding:16px;border-radius:8px;background-color:#16283D;"
                    f"border-left:4px solid {badge_color}'>"
                    f"<span style='color:{badge_color};font-weight:bold;font-family:monospace'>{badge_text}</span><br>"
                    f"<span style='font-size:22px;font-weight:bold'>{r['class']}</span><br>"
                    f"<span style='color:#9DB0BD'>Confidence: {r['confidence']:.2f}%</span>"
                    f"</div>", unsafe_allow_html=True
                )
                c1, c2, c3 = st.columns(3)
                c1.metric("Inference", f"{r['inference_ms']:.2f} ms")
                c2.metric("Explanation", f"{r['explain_ms']:.2f} ms")
                c3.metric("Total Latency", f"{r['total_ms']:.2f} ms")

                if r["is_attack"] and r.get("ai_message"):
                    st.markdown(
                        f"<div class='ai-note'>🤖 <b>AI Analyst Note</b><br>{r['ai_message']}</div>",
                        unsafe_allow_html=True
                    )
        else:
            placeholder_flow.info("Toggle **Start live monitoring** in the sidebar to begin.")

    else:
        st.markdown("#### 🧪 Manual Flow Testing")
        preset = None
        pcol1, pcol2 = st.columns(2)
        if pcol1.button("Load Normal Example"):
            preset = "normal"
        if pcol2.button("Load Attack Example"):
            preset = "attack"

        preset_vals = {}
        if preset == "normal" and len(sample_flows):
            preset_vals = sample_flows.iloc[0].to_dict()
        elif preset == "attack" and len(sample_flows) > 2:
            preset_vals = sample_flows.iloc[2].to_dict()

        with st.form("manual_form"):
            user_input = {}
            cols = st.columns(3)
            for i, feat in enumerate(selected_features):
                default_val = float(preset_vals.get(feat, 0.0))
                user_input[feat] = cols[i % 3].number_input(feat, value=default_val, format="%.4f")
            submitted = st.form_submit_button("🚀 Analyze Flow", type="primary")

        if submitted:
            row = pd.DataFrame([{col: user_input.get(col, 0) for col in all_feature_columns}])
            result = run_inference(row)
            log_result(result, source_label="manual")
            r = st.session_state.last_result
            badge_color = "#F0503C" if r["is_attack"] else "#2ECC71"
            st.markdown(
                f"<div style='padding:16px;border-radius:8px;background-color:#16283D;"
                f"border-left:4px solid {badge_color}'>"
                f"<span style='font-size:22px;font-weight:bold'>{r['class']}</span> "
                f"<span style='color:#9DB0BD'>({r['confidence']:.2f}% confidence)</span><br>"
                f"<span style='color:#9DB0BD'>Total latency: {r['total_ms']:.2f} ms</span>"
                f"</div>", unsafe_allow_html=True
            )
            if r["is_attack"] and r.get("ai_message"):
                st.markdown(
                    f"<div class='ai-note'>🤖 <b>AI Analyst Note</b><br>{r['ai_message']}</div>",
                    unsafe_allow_html=True
                )

    if st.session_state.last_result:
        st.markdown("#### 🔬 Why this prediction — SHAP Explanation")
        r = st.session_state.last_result
        top_feats, vals, order = get_top_shap_features(r, top_k=10)
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#0D1B2A")
        ax.set_facecolor("#0D1B2A")
        colors = ["#F0503C" if v > 0 else "#1B9AAA" for v in vals[order]]
        ax.barh([selected_features[i] for i in order][::-1], vals[order][::-1], color=colors[::-1])
        ax.tick_params(colors="#E8EEF2")
        ax.set_xlabel("SHAP value (impact on prediction)", color="#E8EEF2")
        for spine in ax.spines.values():
            spine.set_color("#29405A")
        st.pyplot(fig)

with right:
    st.markdown("#### 🚨 Live Alert Log")
    if st.session_state.alert_log:
        for entry in st.session_state.alert_log:
            css_class = "alert-row-attack" if entry["is_attack"] else "alert-row-benign"
            icon = "⚠️" if entry["is_attack"] else "✓"
            st.markdown(
                f"<div class='{css_class}'>{icon} <b>{entry['time']}</b> — {entry['class']} "
                f"<span style='color:#9DB0BD'>({entry['confidence']}, {entry['latency_ms']} ms, {entry['source']})</span></div>",
                unsafe_allow_html=True
            )
    else:
        st.info("No flows analyzed yet this session.")

    st.markdown("#### 📈 Latency Trend")
    if len(st.session_state.latencies) > 1:
        st.line_chart(pd.DataFrame({"latency_ms": st.session_state.latencies[-40:]}))
    else:
        st.caption("Appears after a few flows are analyzed.")

# ─────────────────────────────────────────────────────────────────────────
# AUTO-REFRESH FOR LIVE MODE
# ─────────────────────────────────────────────────────────────────────────
if mode.startswith("🔴") and st.session_state.live_mode:
    time.sleep(refresh_rate)
    st.rerun()
