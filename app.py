import time
from datetime import datetime
from pathlib import Path

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
    .alert-row-attack { background-color: rgba(240,80,60,0.15); padding: 6px; border-radius: 4px; margin-bottom: 4px; }
    .alert-row-benign { background-color: rgba(27,154,170,0.08); padding: 6px; border-radius: 4px; margin-bottom: 4px; }
    .block-container { padding-top: 1.5rem; }
    hr { border-color: #29405A; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

ARTIFACT_DIR = Path(__file__).parent / "model_artifacts"


# ─────────────────────────────────────────────────────────────────────────
# LOAD MODEL ARTIFACTS  (no sample_flows.csv needed)
# ─────────────────────────────────────────────────────────────────────────
def _load(name):
    """Load one artifact and report exactly which file failed and why."""
    path = ARTIFACT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{name} not found at {path}")
    size = path.stat().st_size
    try:
        return joblib.load(path)
    except BaseException as e:  # includes MemoryError
        raise RuntimeError(
            f"{name} ({size / 1e6:.2f} MB) failed to load -> "
            f"{type(e).__name__}: {e!r}"
        ) from e


@st.cache_resource
def load_artifacts():
    model = _load("xgboost_xids_model_pso.pkl")
    scaler = _load("scaler.pkl")
    le = _load("label_encoder.pkl")
    selected_features = list(_load("pso_selected_features.pkl"))
    selected_mask = np.asarray(_load("pso_selected_mask.pkl"))
    all_feature_columns = list(_load("all_feature_columns.pkl"))

    # Model was trained on GPU; Streamlit Cloud is CPU only
    try:
        model.set_params(device="cpu")
    except Exception:
        pass

    explainer = shap.TreeExplainer(model)
    return model, scaler, le, selected_features, selected_mask, all_feature_columns, explainer


try:
    (model, scaler, le, selected_features, selected_mask,
     all_feature_columns, explainer) = load_artifacts()
    ARTIFACTS_OK = True
except BaseException as e:
    ARTIFACTS_OK = False
    LOAD_ERROR = f"{type(e).__name__}: {e}"


# ─────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "flows_analyzed": 0,
        "alerts_triggered": 0,
        "latencies": [],
        "alert_log": [],
        "last_result": None,
        "live_mode": False,
        "start_time": time.time(),
        "manual_defaults": {},
        "manual_version": 0,
        "uploaded_df": None,
        "rng_seed": int(time.time()) % 100000,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ─────────────────────────────────────────────────────────────────────────
# FLOW SOURCES
# ─────────────────────────────────────────────────────────────────────────
def synthetic_flow(rng):
    """
    Random demo flow built from the scaler's own statistics (mean / std of the
    training data). NOT real traffic — it only exercises the pipeline.
    """
    z = rng.normal(size=len(all_feature_columns)) * rng.choice([0.5, 1.0, 2.5])
    raw = z * scaler.scale_ + scaler.mean_
    raw = np.maximum(raw, 0)
    return pd.DataFrame([raw], columns=all_feature_columns)


def next_flow(source):
    if source == "Uploaded CSV" and st.session_state.uploaded_df is not None:
        return st.session_state.uploaded_df.sample(1).reset_index(drop=True)
    rng = np.random.default_rng()
    return synthetic_flow(rng)


# ─────────────────────────────────────────────────────────────────────────
# INFERENCE + EXPLANATION
# ─────────────────────────────────────────────────────────────────────────
def class_shap(sv, pred_idx):
    """Return SHAP values (n_features,) for the predicted class across shap versions."""
    if isinstance(sv, list):
        return np.array(sv[pred_idx])[0]
    sv = np.array(sv)
    if sv.ndim == 3:                 # (n_samples, n_features, n_classes)
        return sv[0, :, pred_idx]
    return sv[0]


def run_inference(raw_row_df):
    row_scaled = scaler.transform(raw_row_df[all_feature_columns])
    row_selected = row_scaled[:, selected_mask]

    t0 = time.perf_counter()
    pred = model.predict(row_selected)
    pred_proba = model.predict_proba(row_selected)
    t1 = time.perf_counter()
    shap_values = explainer.shap_values(row_selected)
    t2 = time.perf_counter()

    pred_idx = int(pred[0])
    return {
        "class": str(le.inverse_transform(pred)[0]),
        "confidence": float(pred_proba[0][pred_idx] * 100),
        "inference_ms": (t1 - t0) * 1000,
        "explain_ms": (t2 - t1) * 1000,
        "total_ms": (t2 - t0) * 1000,
        "shap_vals": class_shap(shap_values, pred_idx),
    }


def is_attack_class(name):
    return name.strip().lower() != "benign"


def log_result(result, source_label="live"):
    st.session_state.flows_analyzed += 1
    st.session_state.latencies.append(result["total_ms"])
    attack = is_attack_class(result["class"])
    if attack:
        st.session_state.alerts_triggered += 1
    st.session_state.alert_log.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "class": result["class"],
        "confidence": f'{result["confidence"]:.1f}%',
        "latency_ms": f'{result["total_ms"]:.2f}',
        "is_attack": attack,
        "source": source_label,
    })
    st.session_state.alert_log = st.session_state.alert_log[:25]
    st.session_state.last_result = result


def reset_stats():
    st.session_state.flows_analyzed = 0
    st.session_state.alerts_triggered = 0
    st.session_state.latencies = []
    st.session_state.alert_log = []
    st.session_state.last_result = None


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
        "Model artifacts could not be loaded. Make sure the `model_artifacts/` folder "
        "(next to app.py) contains: xgboost_xids_model_pso.pkl, scaler.pkl, "
        "label_encoder.pkl, pso_selected_features.pkl, pso_selected_mask.pkl, "
        "all_feature_columns.pkl.\n\n"
        f"Details: {LOAD_ERROR}"
    )
    st.write("Files found in model_artifacts/:")
    st.code("\n".join(
        f"{f.name:40s} {f.stat().st_size / 1e6:10.3f} MB"
        for f in sorted(ARTIFACT_DIR.glob("*"))
    ) if ARTIFACT_DIR.exists() else f"Folder not found: {ARTIFACT_DIR}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR — CONTROLS
# ─────────────────────────────────────────────────────────────────────────
refresh_rate = 2.0
source = "Synthetic demo flows"

with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    mode = st.radio("Mode", ["🔴 Live Traffic Simulation", "🧪 Manual Flow Testing"])
    st.markdown("---")

    if mode.startswith("🔴"):
        source = st.radio("Flow source", ["Synthetic demo flows", "Uploaded CSV"])

        if source == "Uploaded CSV":
            up = st.file_uploader("Upload raw (unscaled) flows CSV", type="csv")
            if up is not None:
                df_up = pd.read_csv(up)
                df_up.columns = df_up.columns.str.strip()
                df_up = df_up.replace([np.inf, -np.inf], np.nan).dropna()
                missing = [c for c in all_feature_columns if c not in df_up.columns]
                if missing:
                    st.error(f"CSV is missing {len(missing)} required columns "
                             f"(e.g. {missing[:3]}).")
                    st.session_state.uploaded_df = None
                else:
                    st.session_state.uploaded_df = df_up[all_feature_columns]
                    st.success(f"Loaded {len(df_up):,} flows.")
            if st.session_state.uploaded_df is None:
                st.caption("No valid CSV yet — falling back to synthetic flows.")
        else:
            st.caption("Random flows generated from the training-data statistics. "
                       "They exercise the pipeline but are not real traffic, so the "
                       "predicted classes will not be meaningful.")

        refresh_rate = st.slider("Simulated flow interval (seconds)", 1.0, 5.0, 2.0, 0.5)
        st.session_state.live_mode = st.toggle("▶ Start live monitoring",
                                               value=st.session_state.live_mode)
    else:
        st.session_state.live_mode = False
        st.caption("Set feature values and classify a single custom flow on demand.")

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.caption("**Classifier:** XGBoost")
    st.caption("**Explainer:** SHAP TreeExplainer")
    st.caption(f"**Features used:** {len(selected_features)} (PSO-selected)")
    st.caption(f"**Classes:** {len(le.classes_)}")

    if st.button("🔄 Reset session stats"):
        reset_stats()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────
# TOP METRICS ROW (placeholder, filled after the flow is processed so the
# counters are always up to date)
# ─────────────────────────────────────────────────────────────────────────
metrics_ph = st.container()
st.markdown("<hr>", unsafe_allow_html=True)


def render_top_metrics():
    avg_latency = float(np.mean(st.session_state.latencies)) if st.session_state.latencies else 0.0
    alert_rate = (st.session_state.alerts_triggered / st.session_state.flows_analyzed * 100
                  if st.session_state.flows_analyzed else 0.0)
    with metrics_ph:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Flows Analyzed", f"{st.session_state.flows_analyzed:,}")
        m2.metric("Alerts Triggered", f"{st.session_state.alerts_triggered:,}")
        m3.metric("Avg Latency", f"{avg_latency:.2f} ms")
        m4.metric("Alert Rate", f"{alert_rate:.1f}%")


def render_result_card(r, show_latency_breakdown=True):
    attack = is_attack_class(r["class"])
    color = "#F0503C" if attack else "#2ECC71"
    text = "⚠ THREAT DETECTED" if attack else "✓ NORMAL TRAFFIC"
    st.markdown(
        f"<div style='padding:16px;border-radius:8px;background-color:#16283D;"
        f"border-left:4px solid {color}'>"
        f"<span style='color:{color};font-weight:bold;font-family:monospace'>{text}</span><br>"
        f"<span style='font-size:22px;font-weight:bold'>{r['class']}</span><br>"
        f"<span style='color:#9DB0BD'>Confidence: {r['confidence']:.2f}%</span>"
        f"</div>", unsafe_allow_html=True
    )
    if show_latency_breakdown:
        c1, c2, c3 = st.columns(3)
        c1.metric("Inference", f"{r['inference_ms']:.2f} ms")
        c2.metric("Explanation", f"{r['explain_ms']:.2f} ms")
        c3.metric("Total Latency", f"{r['total_ms']:.2f} ms")


# ─────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────
left, right = st.columns([1.3, 1])

with left:
    if mode.startswith("🔴"):
        st.markdown("#### 📡 Incoming Flow")
        if st.session_state.live_mode:
            result = run_inference(next_flow(source))
            log_result(result, source_label="live")

        if st.session_state.last_result:
            render_result_card(st.session_state.last_result)
        else:
            st.info("Toggle **Start live monitoring** in the sidebar to begin the simulation.")

    else:
        st.markdown("#### 🧪 Manual Flow Testing")
        if st.button("🎲 Load random example flow"):
            flow = synthetic_flow(np.random.default_rng())
            st.session_state.manual_defaults = flow.iloc[0].to_dict()
            st.session_state.manual_version += 1

        base = dict(zip(all_feature_columns, scaler.mean_))
        base.update(st.session_state.manual_defaults)
        ver = st.session_state.manual_version

        with st.form("manual_form"):
            user_input = {}
            cols = st.columns(3)
            for i, feat in enumerate(selected_features):
                user_input[feat] = cols[i % 3].number_input(
                    feat, value=float(base.get(feat, 0.0)),
                    format="%.4f", key=f"inp_{feat}_{ver}")
            submitted = st.form_submit_button("🚀 Analyze Flow", type="primary")

        if submitted:
            row_vals = {c: base.get(c, 0.0) for c in all_feature_columns}
            row_vals.update(user_input)
            result = run_inference(pd.DataFrame([row_vals]))
            log_result(result, source_label="manual")
            render_result_card(result, show_latency_breakdown=False)

    # SHAP explanation for the current result
    if st.session_state.last_result:
        st.markdown("#### 🔬 Why this prediction — SHAP Explanation")
        vals = st.session_state.last_result["shap_vals"]
        order = np.argsort(np.abs(vals))[-10:][::-1]
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#0D1B2A")
        ax.set_facecolor("#0D1B2A")
        colors = ["#F0503C" if v > 0 else "#1B9AAA" for v in vals[order]]
        ax.barh([selected_features[i] for i in order][::-1],
                vals[order][::-1], color=colors[::-1])
        ax.tick_params(colors="#E8EEF2")
        ax.set_xlabel("SHAP value (impact on prediction)", color="#E8EEF2")
        for spine in ax.spines.values():
            spine.set_color("#29405A")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with right:
    st.markdown("#### 🚨 Live Alert Log")
    if st.session_state.alert_log:
        for entry in st.session_state.alert_log:
            css = "alert-row-attack" if entry["is_attack"] else "alert-row-benign"
            icon = "⚠️" if entry["is_attack"] else "✓"
            st.markdown(
                f"<div class='{css}'>{icon} <b>{entry['time']}</b> — {entry['class']} "
                f"<span style='color:#9DB0BD'>({entry['confidence']}, "
                f"{entry['latency_ms']} ms, {entry['source']})</span></div>",
                unsafe_allow_html=True)
    else:
        st.info("No flows analyzed yet this session.")

    st.markdown("#### 📈 Latency Trend")
    if len(st.session_state.latencies) > 1:
        st.line_chart(pd.DataFrame({"latency_ms": st.session_state.latencies[-40:]}))
    else:
        st.caption("Latency chart appears after a few flows are analyzed.")


render_top_metrics()

# ─────────────────────────────────────────────────────────────────────────
# AUTO-REFRESH LOOP FOR LIVE MODE
# ─────────────────────────────────────────────────────────────────────────
if mode.startswith("🔴") and st.session_state.live_mode:
    time.sleep(refresh_rate)
    st.rerun()
