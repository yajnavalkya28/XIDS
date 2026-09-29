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
    .status-active {
        color: #2ECC71; font-weight: bold; font-family: 'Consolas', monospace;
    }
    .alert-row-attack { background-color: rgba(240,80,60,0.15); padding: 6px; border-radius: 4px; }
    .alert-row-benign { background-color: rgba(27,154,170,0.08); padding: 6px; border-radius: 4px; }
    .block-container { padding-top: 1.5rem; }
    hr { border-color: #29405A; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

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
    # Small bundled sample of real (unlabeled-at-inference) test flows used to
    # simulate a live traffic feed. See README for how this file is generated.
    return pd.read_csv("model_artifacts/sample_flows.csv")


try:
    model, scaler, le, selected_features, selected_mask, all_feature_columns, explainer = load_artifacts()
    sample_flows = load_sample_flows()
    ARTIFACTS_OK = True
except Exception as e:
    ARTIFACTS_OK = False
    LOAD_ERROR = str(e)

# ─────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────
if "flows_analyzed" not in st.session_state:
    st.session_state.flows_analyzed = 0
if "alerts_triggered" not in st.session_state:
    st.session_state.alerts_triggered = 0
if "latencies" not in st.session_state:
    st.session_state.latencies = []
if "alert_log" not in st.session_state:
    st.session_state.alert_log = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "live_mode" not in st.session_state:
    st.session_state.live_mode = False
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()


def run_inference(raw_row_df):
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


def log_result(result, source_label="live"):
    st.session_state.flows_analyzed += 1
    st.session_state.latencies.append(result["total_ms"])
    is_attack = result["class"].strip().lower() != "benign"
    if is_attack:
        st.session_state.alerts_triggered += 1
    st.session_state.alert_log.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "class": result["class"],
        "confidence": f'{result["confidence"]:.1f}%',
        "latency_ms": f'{result["total_ms"]:.2f}',
        "is_attack": is_attack,
        "source": source_label,
    })
    st.session_state.alert_log = st.session_state.alert_log[:25]
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
    st.caption("Explainable AI-based Intrusion Detection · XGBoost + PSO Feature Selection + SHAP")
with header_r:
    st.markdown(
        f"<div style='text-align:right'>"
        f"<span class='status-active'>● SYSTEM ACTIVE</span><br>"
        f"<span style='color:#9DB0BD;font-family:monospace'>Uptime {h:02d}:{m:02d}:{s:02d}</span>"
        f"</div>", unsafe_allow_html=True
    )
st.markdown("<hr>", unsafe_allow_html=True)

if not ARTIFACTS_OK:
    st.error(
        "Model artifacts not found. Make sure `model_artifacts/` contains all "
        "required .pkl files and sample_flows.csv (see README).\n\n"
        f"Details: {LOAD_ERROR}"
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR — CONTROLS
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    mode = st.radio("Mode", ["🔴 Live Traffic Simulation", "🧪 Manual Flow Testing"])
    st.markdown("---")

    if mode.startswith("🔴"):
        refresh_rate = st.slider("Simulated flow interval (seconds)", 1.0, 5.0, 2.0, 0.5)
        st.session_state.live_mode = st.toggle("▶ Start live monitoring", value=st.session_state.live_mode)
        st.caption("Pulls a random flow from a held-out real traffic sample every interval, "
                   "runs it through the full detect + explain pipeline.")
    else:
        st.session_state.live_mode = False
        st.caption("Manually set feature values and classify a single custom flow on demand.")

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.caption(f"**Classifier:** XGBoost")
    st.caption(f"**Explainer:** SHAP TreeExplainer")
    st.caption(f"**Features used:** {len(selected_features)} (PSO-selected)")
    st.caption(f"**Classes:** {len(le.classes_)}")

    if st.button("🔄 Reset session stats"):
        for k in ["flows_analyzed", "alerts_triggered", "latencies", "alert_log", "last_result"]:
            st.session_state[k] = 0 if "analyzed" in k or "triggered" in k else ([] if isinstance(st.session_state[k], list) else None)
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────
# TOP METRICS ROW
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
            is_attack = r["class"].strip().lower() != "benign"
            badge_color = "#F0503C" if is_attack else "#2ECC71"
            badge_text = "⚠ THREAT DETECTED" if is_attack else "✓ NORMAL TRAFFIC"

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
        else:
            placeholder_flow.info("Toggle **Start live monitoring** in the sidebar to begin the simulation.")

    else:
        st.markdown("#### 🧪 Manual Flow Testing")
        preset = None
        pcol1, pcol2 = st.columns(2)
        if pcol1.button("Load Normal Example"):
            preset = "normal"
        if pcol2.button("Load Attack Example"):
            preset = "attack"

        defaults = {}
        if preset == "normal":
            defaults = sample_flows.iloc[0].to_dict() if len(sample_flows) else {}
        elif preset == "attack":
            defaults = sample_flows.iloc[min(1, len(sample_flows)-1)].to_dict() if len(sample_flows) else {}

        with st.form("manual_form"):
            user_input = {}
            cols = st.columns(3)
            for i, feat in enumerate(selected_features):
                default_val = float(defaults.get(feat, 0.0))
                user_input[feat] = cols[i % 3].number_input(feat, value=default_val, format="%.4f")
            submitted = st.form_submit_button("🚀 Analyze Flow", type="primary")

        if submitted:
            row = pd.DataFrame([{col: user_input.get(col, 0) for col in all_feature_columns}])
            result = run_inference(row)
            log_result(result, source_label="manual")
            r = result
            is_attack = r["class"].strip().lower() != "benign"
            badge_color = "#F0503C" if is_attack else "#2ECC71"
            st.markdown(
                f"<div style='padding:16px;border-radius:8px;background-color:#16283D;"
                f"border-left:4px solid {badge_color}'>"
                f"<span style='font-size:22px;font-weight:bold'>{r['class']}</span> "
                f"<span style='color:#9DB0BD'>({r['confidence']:.2f}% confidence)</span><br>"
                f"<span style='color:#9DB0BD'>Total latency: {r['total_ms']:.2f} ms</span>"
                f"</div>", unsafe_allow_html=True
            )

    # SHAP explanation for whichever result is current
    if st.session_state.last_result:
        st.markdown("#### 🔬 Why this prediction — SHAP Explanation")
        r = st.session_state.last_result
        sv = r["shap_values"]
        sv_row = sv[0] if isinstance(sv, list) else sv
        if isinstance(sv, list):
            pred_class_idx = list(le.classes_).index(r["class"])
            vals = np.array(sv[pred_class_idx])[0]
        else:
            vals = np.array(sv)[0]

        order = np.argsort(np.abs(vals))[-10:][::-1]
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
        st.caption("Latency chart appears after a few flows are analyzed.")

# ─────────────────────────────────────────────────────────────────────────
# AUTO-REFRESH LOOP FOR LIVE MODE
# ─────────────────────────────────────────────────────────────────────────
if mode.startswith("🔴") and st.session_state.live_mode:
    time.sleep(refresh_rate)
    st.rerun()
