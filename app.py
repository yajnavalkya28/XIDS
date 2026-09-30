import html
import os
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

GROQ_MODEL = "openai/gpt-oss-20b"

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
    all_feature_columns = [str(c).strip() for c in joblib.load("model_artifacts/all_feature_columns.pkl")]
    explainer = shap.TreeExplainer(model)

    # Optional: training medians used to fill missing columns (better than 0).
    medians = None
    if os.path.exists("model_artifacts/feature_medians.pkl"):
        try:
            medians = pd.Series(joblib.load("model_artifacts/feature_medians.pkl"))
            medians.index = [str(c).strip() for c in medians.index]
        except Exception:
            medians = None

    return model, scaler, le, selected_features, selected_mask, all_feature_columns, explainer, medians


BENIGN_FILE = "model_artifacts/benign_flows.csv"   # ✅ Normal button reads this file
ATTACK_FILE = "model_artifacts/attack_flows.csv"   # 🚨 Attack button reads this file


@st.cache_data
def load_preset_file(path, default_label):
    """Read one preset CSV -> model feature columns (+ Label). Missing columns are filled with 0."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    label = df["Label"].astype(str).str.strip() if "Label" in df.columns else pd.Series([default_label] * len(df))
    for col in [c for c in all_feature_columns if c not in df.columns]:
        df[col] = 0.0
    out = df[all_feature_columns].copy()
    out["Label"] = label.values
    return out


@st.cache_data
def load_sample_flows():
    """Used by Live mode. Prefers the two separate files, falls back to sample_flows.csv."""
    if os.path.exists(BENIGN_FILE) and os.path.exists(ATTACK_FILE):
        return pd.concat([load_preset_file(BENIGN_FILE, "Benign"),
                          load_preset_file(ATTACK_FILE, "Attack")], ignore_index=True)
    df = pd.read_csv("model_artifacts/sample_flows.csv")
    df.columns = df.columns.str.strip()
    label_col = df["Label"] if "Label" in df.columns else None
    for col in [c for c in all_feature_columns if c not in df.columns]:
        df[col] = 0.0
    out = df[all_feature_columns].copy()
    if label_col is not None:
        out["Label"] = label_col.astype(str).str.strip().values
    return out


try:
    (model, scaler, le, selected_features, selected_mask,
     all_feature_columns, explainer, medians) = load_artifacts()
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

# ─────────────────────────────────────────────────────────────────────────
# ATTACK KNOWLEDGE BASE  (makes every explanation specific to the attack type)
# Matching is by keywords in the lower-cased class name, first match wins,
# so more specific entries come before generic ones (e.g. "ddos" before "dos").
# ─────────────────────────────────────────────────────────────────────────
ATTACK_PROFILES = [
    (("slowbody", "rudy"), {
        "family": "Slow POST DoS (R-U-Dead-Yet / slow body)",
        "mechanism": "Sends an HTTP POST that declares a huge Content-Length, then trickles the body a few bytes at a time so the server keeps the connection and worker busy.",
        "signature": "very long flow duration, few packets, tiny forward payloads, long inter-arrival gaps and large idle times.",
        "action": "set strict request-body timeouts and minimum data rates, cap connections per IP, and front the app with a reverse proxy or WAF.",
    }),
    (("slowread",), {
        "family": "Slow Read DoS",
        "mechanism": "Requests a large resource, then reads the response extremely slowly by advertising a tiny TCP receive window, keeping server sockets and buffers occupied.",
        "signature": "long-lived flows, small forward requests, large backward data delivered slowly, very small initial window sizes and long idle gaps.",
        "action": "enforce minimum read-rate timeouts, limit concurrent connections per client, and tune server socket timeouts.",
    }),
    (("slowloris", "slowhttp", "slowheaders"), {
        "family": "Slow-header DoS (Slowloris family)",
        "mechanism": "Opens many connections and sends incomplete HTTP headers very slowly so the server keeps every worker slot reserved.",
        "signature": "very long flow durations, very few packets and bytes, large idle times, long inter-arrival gaps and tiny segments.",
        "action": "shorten header timeouts, cap connections per source IP, and put a reverse proxy or WAF in front of the web server.",
    }),
    (("heartbleed",), {
        "family": "Heartbleed (OpenSSL memory disclosure)",
        "mechanism": "Malformed TLS heartbeat requests trick a vulnerable OpenSSL server into returning chunks of process memory.",
        "signature": "long TLS session with a large backward payload relative to the small forward request, repeated heartbeat exchanges.",
        "action": "patch OpenSSL immediately, rotate TLS private keys and certificates, and invalidate active sessions and credentials.",
    }),
    (("hulk",), {
        "family": "HTTP flood DoS (HULK)",
        "mechanism": "Generates large volumes of unique, obfuscated HTTP GET requests so caches cannot absorb them and the web server is overloaded.",
        "signature": "short-to-medium flows with several small forward packets, PSH flags, repeated request bursts and a high request rate from one source.",
        "action": "rate-limit HTTP requests per client, enable WAF bot rules, and check web-server CPU and connection queues.",
    }),
    (("goldeneye",), {
        "family": "HTTP DoS (GoldenEye)",
        "mechanism": "Sends HTTP keep-alive requests with randomized headers and no-cache directives to bypass caching and tie up server resources.",
        "signature": "keep-alive style flows, repeated PSH-flagged requests, moderate durations with regular inter-arrival timing.",
        "action": "limit keep-alive requests per connection, enforce per-IP request quotas, and block the client via WAF.",
    }),
    (("ddossim",), {
        "family": "Application-layer DDoS (DDOSIM simulator)",
        "mechanism": "Simulates many legitimate-looking clients opening full TCP connections and sending HTTP requests to exhaust server connection and application capacity.",
        "signature": "completed TCP handshakes from many sources, short flows with a few request packets, regular timing, modest packet sizes.",
        "action": "apply per-source connection and request limits, enable SYN/connection-rate protection, and scale or scrub at the edge.",
    }),
    (("hoic", "loic"), {
        "family": "HTTP flood DDoS (HOIC / LOIC-HTTP)",
        "mechanism": "A crowd of hosts runs LOIC/HOIC to flood the target with high volumes of HTTP requests, often with randomized headers to defeat simple filters.",
        "signature": "many short flows with repeated small HTTP request packets, PSH flags, high packet rate and little variation in size.",
        "action": "rate-limit and challenge HTTP clients at the WAF/CDN, block offending source ranges, and enable upstream DDoS scrubbing.",
    }),
    (("-dns", "-ntp", "-ldap", "-mssql", "-netbios", "-snmp", "-tftp"), {
        "family": "Reflection / amplification DDoS",
        "mechanism": "The attacker sends small spoofed queries to public servers (DNS, NTP, LDAP, MSSQL, NetBIOS, SNMP or TFTP) that reply with much larger responses toward the victim.",
        "signature": "one-directional UDP flows, large response-sized packets with little or no return traffic, very short durations and very high packet and byte rates.",
        "action": "block the abused UDP service port at the edge, apply ingress filtering and rate limits, and request upstream scrubbing; confirm the victim is not running an open resolver.",
    }),
    (("syn",), {
        "family": "SYN flood DDoS",
        "mechanism": "Floods the target with TCP SYN packets, often with spoofed sources, filling the half-open connection table so real clients cannot connect.",
        "signature": "tiny one-to-few-packet flows, SYN flag set, no payload, no completed handshake and very high packet rate.",
        "action": "enable SYN cookies, reduce SYN-RECEIVED timeouts, apply SYN rate limiting and upstream filtering.",
    }),
    (("udp",), {
        "family": "UDP flood DDoS",
        "mechanism": "Saturates the target's bandwidth or a UDP service (including game-lag tools) with high volumes of UDP datagrams.",
        "signature": "one-directional UDP flows, uniform packet sizes, extremely high packet and byte rates, very short inter-arrival times.",
        "action": "rate-limit or drop unneeded UDP at the edge, apply ACLs for the targeted port, and use upstream scrubbing.",
    }),
    (("ddos", "dos"), {
        "family": "Denial of Service / DDoS flood",
        "mechanism": "Overwhelms the service with excessive packets, requests or connections to exhaust bandwidth or capacity.",
        "signature": "very short durations, high packet and byte rates, small or uniform packet sizes and very low inter-arrival times.",
        "action": "rate-limit or blackhole the offending sources, enable upstream scrubbing, and check load-balancer and firewall capacity.",
    }),
    (("portscan", "port scan"), {
        "family": "Reconnaissance (port scan)",
        "mechanism": "The scanner probes many ports to discover open services, typically with one crafted SYN per port.",
        "signature": "single-packet or two-packet flows, near-zero duration, SYN flag with no payload, RST or absent replies, window sizes typical of scanning tools.",
        "action": "identify and block the scanning source, review which ports responded, and tighten firewall exposure.",
    }),
    (("ftp",), {
        "family": "FTP brute force (credential guessing)",
        "mechanism": "An automated tool repeatedly attempts FTP logins with many username/password combinations.",
        "signature": "many short, similar sessions with small forward packets, PSH flags per login attempt, small consistent reply sizes and regular timing.",
        "action": "lock or throttle the source after failed logins, enable fail2ban, disable anonymous or plain-password FTP, and audit for successful logins.",
    }),
    (("ssh",), {
        "family": "SSH brute force (credential guessing)",
        "mechanism": "An automated tool repeatedly attempts SSH logins with many credential pairs against the same service.",
        "signature": "repeated short encrypted sessions with a small uniform packet count, PSH flags, consistent window sizes and regular timing between attempts.",
        "action": "block or throttle the source, enforce key-based auth, enable fail2ban, and audit auth logs for any successful login.",
    }),
    (("sql",), {
        "family": "Web attack: SQL injection",
        "mechanism": "Malicious SQL fragments are injected through web request parameters to read or modify the database.",
        "signature": "HTTP flows with unusually large forward payloads (long crafted query strings), PSH-flagged requests and larger-than-normal responses.",
        "action": "review web and database logs for the offending queries, enable parameterized queries and WAF SQLi rules, and check for data exfiltration.",
    }),
    (("xss",), {
        "family": "Web attack: Cross-Site Scripting (XSS)",
        "mechanism": "Script payloads are injected into web parameters so they execute in other users' browsers.",
        "signature": "HTTP flows with larger forward payloads containing crafted parameters, repeated similar requests and PSH-flagged request packets.",
        "action": "inspect the targeted endpoints, enforce input sanitization and output encoding, and enable a Content-Security-Policy and WAF rules.",
    }),
    (("webattack", "web attack", "bruteforce", "brute force"), {
        "family": "Web brute force (login guessing over HTTP)",
        "mechanism": "Automated repeated login requests against a web form or endpoint to guess valid credentials.",
        "signature": "many similar short HTTP flows with small POST-style payloads, PSH flags and steady inter-arrival timing.",
        "action": "add rate-limiting or CAPTCHA to login endpoints, lock accounts after repeated failures, and review auth logs for successful logins.",
    }),
    (("bot",), {
        "family": "Botnet command-and-control activity",
        "mechanism": "A compromised host beacons to a C2 server to fetch instructions or exfiltrate data.",
        "signature": "periodic, regular-interval small exchanges, consistent packet sizes and long-lived or repeated sessions to the same endpoint.",
        "action": "isolate the infected host, block the C2 destination, and run endpoint malware scanning and credential rotation.",
    }),
    (("infil",), {
        "family": "Infiltration (post-exploitation activity)",
        "mechanism": "A compromised internal host is used to download tools, move laterally or stage data for exfiltration.",
        "signature": "unusual long or bursty sessions with asymmetric byte volumes and atypical idle/active timing compared with normal traffic.",
        "action": "isolate the host, review its recent connections and processes, and hunt for lateral movement.",
    }),
]

DEFAULT_PROFILE = {
    "family": "Unclassified malicious traffic",
    "mechanism": "Traffic pattern that deviates strongly from normal behavior learned by the model.",
    "signature": "flow statistics that fall outside benign ranges.",
    "action": "investigate the source and destination, review related logs, and consider temporary blocking.",
}


def get_attack_profile(class_name):
    name = str(class_name).lower()
    for keywords, profile in ATTACK_PROFILES:
        if any(k in name for k in keywords):
            return profile
    return DEFAULT_PROFILE


def fmt_val(v):
    try:
        v = float(v)
    except Exception:
        return str(v)
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.4g}"


def build_evidence(result, top_features):
    raw = result["raw_row"].iloc[0]
    lines = []
    for name, shap_val in top_features:
        val = raw[name] if name in raw.index else float("nan")
        direction = "pushes toward" if shap_val > 0 else "pushes away from"
        lines.append(f"- {name} = {fmt_val(val)} (SHAP {shap_val:+.3f}, {direction} {result['class']})")
    return "\n".join(lines)


def fallback_explanation(result, top_features):
    """Rule-based note used if the LLM is unavailable or returns nothing."""
    p = get_attack_profile(result["class"])
    raw = result["raw_row"].iloc[0]
    evid = ", ".join(f"{n}={fmt_val(raw.get(n, 0))}" for n, _ in top_features[:3])
    return (f"{result['class']} ({p['family']}): the strongest evidence is {evid}, "
            f"which matches the typical pattern of {p['signature']} "
            f"Recommended action: {p['action']}")


def _call_groq(prompt):
    kwargs = dict(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=900,   # gpt-oss is a reasoning model: reasoning tokens count toward this limit
    )
    try:
        resp = groq_client.chat.completions.create(**kwargs, reasoning_effort="low")
    except Exception:
        resp = groq_client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def generate_ai_explanation(result, top_features):
    """Attack-specific analyst note. Never called for benign traffic."""
    if groq_client is None:
        return fallback_explanation(result, top_features)

    p = get_attack_profile(result["class"])
    evidence = build_evidence(result, top_features)
    others = ", ".join(f"{n} {c:.1f}%" for n, c in result["top_classes"][1:3])

    prompt = f"""You are a senior SOC analyst writing a dashboard alert note for ONE specific detection.

Detected class: {result['class']}  ({p['family']})
Model confidence: {result['confidence']:.1f}%
Next most likely classes: {others}

Background on this attack type:
- How it works: {p['mechanism']}
- Typical flow signature: {p['signature']}
- Standard response: {p['action']}

Evidence from THIS flow (raw values; times are in microseconds, rates per second):
{evidence}

Write exactly 3 sentences, plain text, no headers, no bullets, max 95 words total:
1. Say what this specific attack is doing and tie it to the 2-3 strongest features above, interpreting their values (e.g. "0.4 ms duration with 3M packets means...").
2. If a next-likely class is within about 15 points of the top one, say why it could be confused with {result['class']}; otherwise say what makes this flow a clear {result['class']} case.
3. Give one concrete action specific to {result['class']}.

Rules: name the attack type explicitly; never write generic phrases like "suspicious traffic" or "anomalous behavior"; do not invent IPs, ports or facts that are not in the evidence."""

    try:
        text = _call_groq(prompt)
        return text if text else fallback_explanation(result, top_features)
    except Exception as e:
        return f"{fallback_explanation(result, top_features)}  (LLM unavailable: {e})"


# ─────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────
defaults = {
    "flows_analyzed": 0,
    "alerts_triggered": 0,
    "latencies": [],
    "alert_log": [],
    "incident_log": [],
    "last_result": None,
    "live_mode": False,
    "start_time": time.time(),
    "batch_results": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def prepare_raw(df):
    """Clean any dataframe into the model's raw feature layout (strip names, numeric, no inf/NaN)."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.reindex(columns=all_feature_columns)
    df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if medians is not None:
        df = df.fillna(medians)
    return df.fillna(0.0)


def run_inference(raw_row_df):
    raw_row_df = prepare_raw(raw_row_df)

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

    order = np.argsort(pred_proba[0])[::-1][:3]
    top_classes = [(str(le.inverse_transform([int(i)])[0]), float(pred_proba[0][i] * 100)) for i in order]

    return {
        "class": predicted_class,
        "confidence": confidence,
        "top_classes": top_classes,
        "inference_ms": (t1 - t0) * 1000,
        "explain_ms": (t2 - t1) * 1000,
        "total_ms": (t2 - t0) * 1000,
        "shap_values": shap_values,
        "pred_idx": int(pred[0]),
        "raw_row": raw_row_df,
    }


def get_top_shap_features(result, top_k=5):
    """Robust to every SHAP output shape (list per class, 3D array, or 2D array)."""
    sv = result["shap_values"]
    pred_idx = result["pred_idx"]

    if isinstance(sv, list):
        vals = np.asarray(sv[pred_idx])[0]
    else:
        sv_arr = np.asarray(sv)
        if sv_arr.ndim == 3:
            vals = sv_arr[0, :, pred_idx]
        else:
            vals = sv_arr[0]

    vals = np.asarray(vals).reshape(-1)
    top_k = min(top_k, len(vals))
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
        with st.spinner("🤖 Generating attack-specific analysis..."):
            ai_message = generate_ai_explanation(result, top_feats)

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
    return result


def render_result_panel(r):
    badge_color = "#F0503C" if r["is_attack"] else "#2ECC71"
    badge_text = "⚠ THREAT DETECTED" if r["is_attack"] else "✓ NORMAL TRAFFIC"
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
            f"<div class='ai-note'>🤖 <b>AI Analyst Note — {html.escape(r['class'])}</b><br>"
            f"{html.escape(r['ai_message'])}</div>",
            unsafe_allow_html=True,
        )
    with st.expander("Class probabilities (top 3)"):
        for name, pct in r["top_classes"]:
            st.write(f"{name}: {pct:.2f}%")


def render_shap_chart(r):
    st.markdown("#### 🔬 Why this prediction — SHAP Explanation")
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
    plt.close(fig)


def render_alert_log(entries):
    for entry in entries:
        css_class = "alert-row-attack" if entry["is_attack"] else "alert-row-benign"
        icon = "⚠️" if entry["is_attack"] else "✓"
        st.markdown(
            f"<div class='{css_class}'>{icon} <b>{entry['time']}</b> — {html.escape(str(entry['class']))} "
            f"<span style='color:#9DB0BD'>({entry['confidence']}, {entry['latency_ms']} ms, {entry['source']})</span></div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────
uptime = int(time.time() - st.session_state.start_time)
h, rem = divmod(uptime, 3600)
m, s = divmod(rem, 60)

header_l, header_r = st.columns([3, 1])
with header_l:
    st.markdown("## 🛡️ XIDS — Real-Time Network Intrusion Monitoring")
    st.caption("XGBoost + PSO Feature Selection + SHAP · attack-specific analyst explanations via Groq")
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
        "⚠ GROQ_API_KEY not found in st.secrets — attack notes will use the built-in rule-based "
        "explanation until you add it (Streamlit Cloud: App settings → Secrets)."
    )

# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    mode = st.radio("Mode", ["🔴 Live Traffic Simulation", "🧪 Manual Flow Testing", "📁 Batch CSV Upload"])
    st.markdown("---")

    if mode.startswith("🔴"):
        refresh_rate = st.slider("Simulated flow interval (seconds)", 1.0, 5.0, 2.0, 0.5)
        st.session_state.live_mode = st.toggle("▶ Start live monitoring", value=st.session_state.live_mode)
    else:
        st.session_state.live_mode = False

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.caption("**Classifier:** XGBoost  |  **Explainer:** SHAP TreeExplainer")
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
# MODE: LIVE SIMULATION
# ─────────────────────────────────────────────────────────────────────────
if mode.startswith("🔴"):
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("#### 📡 Incoming Flow")
        placeholder_flow = st.empty()

        if st.session_state.live_mode:
            feat_cols = [c for c in sample_flows.columns if c != "Label"]
            row = sample_flows[feat_cols].sample(1).reset_index(drop=True)
            result = run_inference(row)
            log_result(result, source_label="live")

        if st.session_state.last_result:
            with placeholder_flow.container():
                render_result_panel(st.session_state.last_result)
        else:
            placeholder_flow.info("Toggle **Start live monitoring** in the sidebar to begin.")

        if st.session_state.last_result:
            render_shap_chart(st.session_state.last_result)

    with right:
        st.markdown("#### 🚨 Live Alert Log")
        if st.session_state.alert_log:
            render_alert_log(st.session_state.alert_log)
        else:
            st.info("No flows analyzed yet this session.")

        st.markdown("#### 📈 Latency Trend")
        if len(st.session_state.latencies) > 1:
            st.line_chart(pd.DataFrame({"latency_ms": st.session_state.latencies[-40:]}))
        else:
            st.caption("Appears after a few flows are analyzed.")

    if st.session_state.live_mode:
        time.sleep(refresh_rate)
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────
# MODE: MANUAL TESTING
# ─────────────────────────────────────────────────────────────────────────
elif mode.startswith("🧪"):
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("#### 🧪 Manual Flow Testing")
        st.caption("Preset buttons run the prediction immediately. Expand 'Custom values' below to type your own numbers.")

        # ── Two separate preset files: one per button ──────────────────────────
        if os.path.exists(BENIGN_FILE):
            benign_rows = load_preset_file(BENIGN_FILE, "Benign")
        else:
            benign_rows = sample_flows[sample_flows["Label"].astype(str).str.lower() == "benign"]
        if os.path.exists(ATTACK_FILE):
            attack_rows = load_preset_file(ATTACK_FILE, "Attack")
        else:
            attack_rows = sample_flows[sample_flows["Label"].astype(str).str.lower() != "benign"]

        st.caption(f"✅ Normal button → `{BENIGN_FILE}` ({len(benign_rows)} rows)   |   "
                   f"🚨 Attack button → `{ATTACK_FILE}` ({len(attack_rows)} rows)")

        attack_types = ["Any attack type"] + sorted(attack_rows["Label"].astype(str).unique().tolist())
        chosen_attack = st.selectbox("Attack type for the attack button (Any = cycles through all types)", attack_types)
        if chosen_attack != "Any attack type":
            attack_rows = attack_rows[attack_rows["Label"].astype(str) == chosen_attack]

        pcol1, pcol2 = st.columns(2)

        if pcol1.button("✅ Normal Example", use_container_width=True):
            if len(benign_rows) == 0:
                st.warning(f"No rows found in {BENIGN_FILE}.")
            else:
                picked = benign_rows.sample(1)
                st.session_state["last_true_label"] = str(picked["Label"].iloc[0])
                result = run_inference(picked.drop(columns=["Label"]).reset_index(drop=True))
                log_result(result, source_label="manual-normal-file")

        if pcol2.button("🚨 Attack Example", use_container_width=True):
            if len(attack_rows) == 0:
                st.warning(f"No rows found in {ATTACK_FILE}.")
            else:
                if chosen_attack == "Any attack type":
                    # cycle through EVERY attack type in turn, so each click shows a different one
                    types_list = sorted(attack_rows["Label"].astype(str).unique().tolist())
                    idx = st.session_state.get("attack_cycle", 0) % len(types_list)
                    st.session_state["attack_cycle"] = idx + 1
                    pool = attack_rows[attack_rows["Label"].astype(str) == types_list[idx]]
                else:
                    pool = attack_rows
                picked = pool.sample(1)
                st.session_state["last_true_label"] = str(picked["Label"].iloc[0])
                result = run_inference(picked.drop(columns=["Label"]).reset_index(drop=True))
                log_result(result, source_label="manual-attack-file")

        with st.expander("📊 Detection report — how well does the model catch each attack type?"):
            st.caption(f"Runs the model over every row of `{ATTACK_FILE}` (no SHAP / AI, so it is fast).")
            if st.button("Run detection report"):
                if not os.path.exists(ATTACK_FILE):
                    st.warning(f"{ATTACK_FILE} not found.")
                else:
                    full = load_preset_file(ATTACK_FILE, "Attack")
                    raw = prepare_raw(full.drop(columns=["Label"]))
                    proba = model.predict_proba(scaler.transform(raw[all_feature_columns])[:, selected_mask])
                    rep = pd.DataFrame({
                        "true_label": full["Label"].astype(str).values,
                        "predicted": le.inverse_transform(proba.argmax(1)),
                        "conf": proba.max(1) * 100,
                    })
                    rep["detected"] = rep["predicted"].str.strip().str.lower() != "benign"
                    table = rep.groupby("true_label").agg(
                        rows=("true_label", "size"),
                        detected_pct=("detected", lambda x: round(x.mean() * 100, 1)),
                        most_common_prediction=("predicted", lambda x: x.mode().iloc[0]),
                        avg_confidence=("conf", lambda x: round(x.mean(), 1)),
                    ).sort_values("detected_pct")
                    st.dataframe(table, use_container_width=True)
                    st.info(f"Overall: {rep['detected'].mean() * 100:.1f}% of attack rows flagged as an attack.")

        with st.expander("✏️ Custom values (advanced)"):
            with st.form("manual_form"):
                user_input = {}
                cols = st.columns(3)
                for i, feat in enumerate(selected_features):
                    user_input[feat] = cols[i % 3].number_input(feat, value=0.0, format="%.4f")
                submitted = st.form_submit_button("🚀 Analyze Custom Flow", type="primary")

            if submitted:
                row = pd.DataFrame([{col: user_input.get(col, np.nan) for col in all_feature_columns}])
                result = run_inference(row)
                log_result(result, source_label="manual-custom")

        if st.session_state.last_result:
            st.markdown("---")
            true_lbl = st.session_state.get("last_true_label")
            if true_lbl:
                st.caption(f"Row loaded from file — true label: **{true_lbl}**")
                if (st.session_state.last_result["class"].strip().lower() == "benign"
                        and true_lbl.strip().lower() != "benign"):
                    st.warning(f"Model missed this one: true label is **{true_lbl}** but it predicted Benign.")
            render_result_panel(st.session_state.last_result)
            render_shap_chart(st.session_state.last_result)

    with right:
        st.markdown("#### 🚨 Recent Results")
        if st.session_state.alert_log:
            render_alert_log(st.session_state.alert_log)
        else:
            st.info("No flows analyzed yet this session.")

# ─────────────────────────────────────────────────────────────────────────
# MODE: BATCH UPLOAD (CSV or Excel)
# ─────────────────────────────────────────────────────────────────────────
else:
    st.markdown("#### 📁 Batch CSV / Excel Upload")
    st.caption(
        "Upload a CSV or Excel file of one or more flows. Columns can be any subset of the model's "
        f"{len(all_feature_columns)} original features (or just the {len(selected_features)} "
        "PSO-selected ones) — missing ones are filled with training medians (or 0). Every row is "
        "classified, and every attack row gets an attack-specific AI explanation + an incident log entry."
    )

    uploaded = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".xlsx"):
                batch_df = pd.read_excel(uploaded)
            else:
                batch_df = pd.read_csv(uploaded)
            batch_df.columns = batch_df.columns.astype(str).str.strip()

            matched = [c for c in all_feature_columns if c in batch_df.columns]
            st.success(f"Loaded {len(batch_df)} row(s). Matched {len(matched)}/{len(all_feature_columns)} model features.")
            if len(matched) == 0:
                st.error("None of the column names match the model's features — every row would be identical. "
                         "Check the column names in your file.")
            st.dataframe(batch_df.head(10), use_container_width=True)

            if st.button("🚀 Classify All Rows", type="primary", disabled=len(matched) == 0):
                results_rows = []
                progress = st.progress(0, text="Processing flows...")
                for i in range(len(batch_df)):
                    row = batch_df.iloc[[i]].drop(columns=["Label", "Sample_Type", "Expected"], errors="ignore")
                    result = run_inference(row)
                    logged = log_result(result, source_label=f"batch-row-{i}")
                    entry = {
                        "row": i,
                        "predicted_class": logged["class"],
                        "confidence_%": round(logged["confidence"], 2),
                        "is_attack": logged["is_attack"],
                        "total_latency_ms": round(logged["total_ms"], 2),
                        "ai_explanation": logged.get("ai_message") or "",
                    }
                    if "Label" in batch_df.columns:
                        entry["true_label"] = str(batch_df.iloc[i]["Label"])
                    results_rows.append(entry)
                    progress.progress((i + 1) / len(batch_df), text=f"Processed {i+1}/{len(batch_df)}")

                st.session_state.batch_results = pd.DataFrame(results_rows)
                progress.empty()

            if st.session_state.batch_results is not None:
                st.markdown("#### Results")
                res_df = st.session_state.batch_results

                def highlight_attack(row):
                    color = "background-color: rgba(240,80,60,0.15)" if row["is_attack"] else ""
                    return [color] * len(row)

                st.dataframe(res_df.style.apply(highlight_attack, axis=1), use_container_width=True)

                n_attacks = int(res_df["is_attack"].sum())
                st.info(f"{n_attacks} attack(s) detected out of {len(res_df)} flows analyzed.")

                st.download_button(
                    "⬇️ Download Batch Results (CSV)",
                    data=res_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"xids_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Could not process this file: {e}")
    else:
        st.info("No file uploaded yet. You can use the `sample_flows.csv` in `model_artifacts/` as a template.")
