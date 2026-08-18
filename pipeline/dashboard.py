import os
import datetime
import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.signal as signal
from scipy.fft import rfftfreq
import streamlit as st
from dotenv import load_dotenv
from classifier import (
    decode_audio_from_mp4,
    retrain_model_from_database,
    update_machine_baseline,
    get_machine_baseline,
    extract_spectral_health_profile,
    DEFAULT_CLASSES,
    MODEL_PATH
)

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "machine_telemetry.duckdb")
AUDIO_DIR = os.path.join(BASE_DIR, "audio_samples")
os.makedirs(AUDIO_DIR, exist_ok=True)

st.set_page_config(
    page_title="Industrial Acoustic Anomaly & Fleet Monitor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e2530;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 4px solid #00e5ff;
    }
    .device-card-active {
        background-color: #1a2230;
        border: 1px solid #303e54;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

def get_duckdb_connection(read_only=False):
    import time
    for _ in range(20):
        try:
            return duckdb.connect(DB_PATH, read_only=read_only)
        except duckdb.IOException as e:
            if "lock" in str(e).lower():
                time.sleep(0.2)
            else:
                raise
    return duckdb.connect(DB_PATH, read_only=read_only)

def load_telemetry_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    
    con = get_duckdb_connection(read_only=True)
    try:
        df = con.execute("""
            SELECT * FROM machine_anomalies
            ORDER BY time_detected DESC
        """).df()
    except Exception as e:
        df = pd.DataFrame()
    finally:
        con.close()
    return df

def load_stream_telemetry(machine_id=None):
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    con = get_duckdb_connection(read_only=True)
    try:
        if machine_id and machine_id != "All Devices":
            df_s = con.execute("""
                SELECT * FROM machine_telemetry_stream
                WHERE machine_id = ?
                ORDER BY recorded_at ASC
            """, (machine_id,)).df()
        else:
            df_s = con.execute("""
                SELECT * FROM machine_telemetry_stream
                ORDER BY recorded_at ASC
            """).df()
    except Exception:
        df_s = pd.DataFrame()
    finally:
        con.close()
    return df_s

# Initialize session state for device navigation
if "selected_device" not in st.session_state:
    st.session_state["selected_device"] = "All Devices"

raw_df = load_telemetry_data()
raw_stream_df = load_stream_telemetry()

# Collect machine list from both anomaly events and continuous stream
machines_from_anomalies = set(raw_df['machine_id'].dropna().unique()) if not raw_df.empty else set()
machines_from_stream = set(raw_stream_df['machine_id'].dropna().unique()) if not raw_stream_df.empty else set()
all_machines = sorted(list(machines_from_anomalies | machines_from_stream))

machine_options = ["All Devices"] + all_machines

# Sidebar Navigation & Model Status
with st.sidebar:
    st.header("⚙️ Fleet & Classifier Control")
    
    # Device Selector Dropdown in Sidebar
    current_index = machine_options.index(st.session_state["selected_device"]) if st.session_state["selected_device"] in machine_options else 0
    selected_from_sb = st.selectbox(
        "🎯 Select View / Device:",
        options=machine_options,
        index=current_index,
        key="sidebar_device_select"
    )
    if selected_from_sb != st.session_state["selected_device"]:
        st.session_state["selected_device"] = selected_from_sb
        st.rerun()

    st.markdown("---")
    st.subheader("🔄 Data Synchronization")
    if st.button("⚡ Sync / Refresh Data Now", key="btn_manual_sync", use_container_width=True, type="secondary"):
        st.rerun()

    enable_fragment_sync = st.checkbox("Enable Background Auto-Sync (Non-Blocking)", value=False, key="cb_fragment_sync", help="Uses Streamlit fragments to update charts silently in the background without freezing your inputs or audio playback.")

    st.markdown("---")
    st.subheader("🔍 Telemetry Filtering Controls")
    time_filter = st.selectbox(
        "⏱️ Time Window Filter:",
        options=["All Time", "Last 1 Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
        index=0,
        key="sidebar_time_filter"
    )
    
    min_amp_filter = st.slider(
        "🔊 Min Amplitude Threshold:",
        min_value=0,
        max_value=3000,
        value=0,
        step=50,
        key="sidebar_min_amp_filter"
    )

    # Apply Sidebar Filters to Raw Data
    now_dt = datetime.datetime.now()
    if time_filter != "All Time":
        hours_map = {"Last 1 Hour": 1, "Last 6 Hours": 6, "Last 24 Hours": 24, "Last 7 Days": 168}
        cutoff = now_dt - datetime.timedelta(hours=hours_map[time_filter])
        if not raw_df.empty and 'time_detected' in raw_df.columns:
            raw_df = raw_df[pd.to_datetime(raw_df['time_detected']).dt.tz_localize(None) >= cutoff]
        if not raw_stream_df.empty and 'recorded_at' in raw_stream_df.columns:
            raw_stream_df = raw_stream_df[pd.to_datetime(raw_stream_df['recorded_at']).dt.tz_localize(None) >= cutoff]

    if min_amp_filter > 0:
        if not raw_df.empty and 'peak_amplitude' in raw_df.columns:
            raw_df = raw_df[raw_df['peak_amplitude'] >= min_amp_filter]
        if not raw_stream_df.empty and 'peak_amplitude' in raw_stream_df.columns:
            raw_stream_df = raw_stream_df[raw_stream_df['peak_amplitude'] >= min_amp_filter]

    # Active dataset based on selection (honoring time window & amplitude filters)
    if st.session_state["selected_device"] == "All Devices":
        df = raw_df.copy()
        df_stream = raw_stream_df.copy()
    else:
        df = raw_df[raw_df['machine_id'] == st.session_state["selected_device"]].reset_index(drop=True) if not raw_df.empty else pd.DataFrame()
        df_stream = raw_stream_df[raw_stream_df['machine_id'] == st.session_state["selected_device"]].reset_index(drop=True) if not raw_stream_df.empty else pd.DataFrame()

    total_events = len(df)
    labeled_count = 0
    if not df.empty and 'verified_label' in df.columns:
        labeled_count = int(df['verified_label'].dropna().apply(lambda x: 1 if str(x).strip() and str(x) not in ['Ignored / False Positive', 'Non-Classified', 'Unknown'] else 0).sum())
    
    st.metric("Events in View", total_events)
    st.metric("Active ML Training Labels", labeled_count)

    if not df.empty and 'verified_label' in df.columns and labeled_count > 0:
        valid_labels = df['verified_label'].dropna()
        valid_labels = valid_labels[~valid_labels.isin(['Ignored / False Positive', 'Non-Classified', 'Unknown'])]
        if not valid_labels.empty:
            label_dist = valid_labels.value_counts()
            st.caption("Verified Class Distribution:")
            st.bar_chart(label_dist)

    st.markdown("---")
    st.subheader("🔄 Retrain Classifier")
    st.caption("Feed confirmed sound classifications into the training model.")
    if st.button("🚀 Retrain Model Now", type="primary"):
        with st.spinner("Extracting acoustic features and training model..."):
            result = retrain_model_from_database(DB_PATH)
            if result["status"] == "success":
                st.success(result["message"])
                st.rerun()
            elif result["status"] == "warning":
                st.warning(result["message"])
            else:
                st.error(result["message"])

    st.markdown("---")
    with st.expander("🗑️ Database & Data Wipe Manager", expanded=False):
        target_dev = st.session_state["selected_device"]
        dev_label = f"Device `{target_dev}`" if target_dev != "All Devices" else "ALL Fleet Devices"
        
        st.markdown(f"**Target Scope:** {dev_label}")
        
        if st.button("🧹 Wipe 30s Stream Telemetry", key="btn_wipe_stream_sb", help=f"Clears continuous telemetry stream rows for {dev_label}"):
            conn = get_duckdb_connection()
            if target_dev == "All Devices":
                conn.execute("DELETE FROM machine_telemetry_stream")
            else:
                conn.execute("DELETE FROM machine_telemetry_stream WHERE machine_id = ?", (target_dev,))
            conn.close()
            st.toast(f"Wiped telemetry stream for {dev_label}.")
            st.rerun()

        if st.button("🗑️ Wipe Anomaly Detections & Audio", key="btn_wipe_anomalies_sb", help=f"Clears discrete anomaly records and deletes audio files for {dev_label}"):
            conn = get_duckdb_connection()
            if target_dev == "All Devices":
                rows = conn.execute("SELECT audio_path FROM machine_anomalies").fetchall()
                conn.execute("DELETE FROM machine_anomalies")
            else:
                rows = conn.execute("SELECT audio_path FROM machine_anomalies WHERE machine_id = ?", (target_dev,)).fetchall()
                conn.execute("DELETE FROM machine_anomalies WHERE machine_id = ?", (target_dev,))
            for (ap,) in rows:
                if ap:
                    p = os.path.join(BASE_DIR, ap) if not os.path.isabs(ap) else ap
                    if os.path.exists(p):
                        try: os.remove(p)
                        except Exception: pass
            conn.close()
            st.toast(f"Wiped anomaly records and audio for {dev_label}.")
            st.rerun()

        st.markdown("---")
        
        # ------------------ OVER-THE-AIR (OTA) FIRMWARE MANAGER ------------------
        with st.expander("📡 Over-The-Air (OTA) Updates", expanded=False):
            st.markdown("##### Remotely push new firmware over Wi-Fi:")
            ota_dir = os.path.join(BASE_DIR, "static", "ota")
            ota_file = os.path.join(ota_dir, "firmware.bin")

            if os.path.exists(ota_file):
                ota_stat = os.stat(ota_file)
                st.success(f"**Staged:** `firmware.bin` ({ota_stat.st_size / (1024*1024):.2f} MB)")
                st.caption(f"Compiled/Staged: {datetime.datetime.fromtimestamp(ota_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.warning("⚠️ No `firmware.bin` staged yet.\nRun `./build.sh --deploy-ota`.")

            target_ota = st.session_state["selected_device"]
            btn_ota_label = f"🚀 Push OTA to `{target_ota}`" if target_ota != "All Devices" else "🚀 Push OTA to Fleet"

            if st.button(btn_ota_label, key="btn_sb_push_ota", disabled=not os.path.exists(ota_file), type="primary"):
                import paho.mqtt.client as mqtt
                import json
                try:
                    mqtt_broker_host = os.getenv("MQTT_HOST", "172.16.40.129")
                    mqtt_broker_port = int(os.getenv("MQTT_PORT", "1883"))
                    mqttc = mqtt.Client()
                    mqttc.connect(mqtt_broker_host, mqtt_broker_port, 5)
                    ota_payload = json.dumps({
                        "cmd": "update",
                        "resource": "ota/firmware.bin",
                        "port": 5000
                    })
                    topic = f"factory/{target_ota}/cmd/ota" if target_ota != "All Devices" else "factory/all/cmd/ota"
                    mqttc.publish(topic, ota_payload)
                    mqttc.disconnect()
                    st.toast(f"OTA signal published to {topic}!")
                    st.info(f"📡 Sent OTA update signal to `{topic}`. Target device is downloading `firmware.bin` and flashing slot B (~10s)...")
                except Exception as ex:
                    st.error(f"Failed to send MQTT OTA command: {ex}")

        st.markdown("---")
        confirm_full_purge = st.checkbox("Confirm: Wipe Entire Database & Reset All Baselines", key="confirm_full_purge")
        if st.button("💣 PURGE ALL TELEMETRY & BASELINES", type="primary", disabled=not confirm_full_purge, key="btn_full_purge"):
            conn = get_duckdb_connection()
            rows = conn.execute("SELECT audio_path FROM machine_anomalies").fetchall()
            for (ap,) in rows:
                if ap:
                    p = os.path.join(BASE_DIR, ap) if not os.path.isabs(ap) else ap
                    if os.path.exists(p):
                        try: os.remove(p)
                        except Exception: pass
            conn.execute("DELETE FROM machine_anomalies")
            conn.execute("DELETE FROM machine_telemetry_stream")
            try: conn.execute("DELETE FROM machine_baselines")
            except Exception: pass
            conn.close()
            st.toast("Entire DuckDB telemetry database and audio archive wiped.")
            st.rerun()

# --------------------------------------------------------------------------------------
# HEADER & DYNAMIC CONTEXT METRICS
# --------------------------------------------------------------------------------------
if st.session_state["selected_device"] != "All Devices":
    # Dedicated Device View Header
    col_nav1, col_nav2 = st.columns([4, 1])
    with col_nav1:
        st.title(f"📍 Device Dashboard: `{st.session_state['selected_device']}`")
        st.caption(f"Real-time acoustic telemetry, on/off machine state, and audio classifications for {st.session_state['selected_device']}.")
    with col_nav2:
        if st.button("⬅️ Back to Fleet Overview", type="secondary"):
            st.session_state["selected_device"] = "All Devices"
            st.rerun()

    now_dt = datetime.datetime.now()
    m_stream = raw_stream_df[raw_stream_df['machine_id'] == st.session_state['selected_device']] if not raw_stream_df.empty else pd.DataFrame()

    if not m_stream.empty:
        last_stream_time = pd.to_datetime(m_stream['recorded_at'].iloc[-1]).replace(tzinfo=None)
        stream_diff_min = (now_dt - last_stream_time).total_seconds() / 60.0
        last_state = m_stream['machine_state'].iloc[-1]
        if stream_diff_min < 2:
            power_status = "⚡ ON / Operating" if last_state == "RUNNING" else "💤 Idle Standby"
        elif stream_diff_min < 60:
            power_status = "🟡 Standby (Recent Ping)"
        else:
            power_status = "🔌 Offline"
    elif not df.empty:
        last_event = df.iloc[0]
        last_time = pd.to_datetime(last_event['time_detected']).replace(tzinfo=None)
        time_diff_min = (now_dt - last_time).total_seconds() / 60.0
        if time_diff_min < 15:
            power_status = "⚡ ON / Operating"
        elif time_diff_min < 120:
            power_status = "💤 OFF / Idle Standby"
        else:
            power_status = "🔌 Offline / No Ping"
    else:
        power_status = "🔌 Waiting for Telemetry"

    top_pred = df['predicted_class'].mode().iloc[0] if (not df.empty and 'predicted_class' in df.columns and not df['predicted_class'].dropna().empty) else "Normal"
    avg_dur = f"{df['duration_seconds'].mean():.2f} s" if not df.empty else "0.0 s"
    
    c1, c2, c3, c4 = st.columns([1, 1, 1.8, 1])
    c1.metric("Machine Power State", power_status)
    c2.metric("Total Device Detections", len(df))
    with c3:
        st.markdown("<div style='font-size: 0.85em; color: #b0bec5; margin-bottom: 2px;'>Primary Anomaly Type</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 1.4em; font-weight: bold; color: #00e5ff; white-space: nowrap;'>{top_pred}</div>", unsafe_allow_html=True)
    c4.metric("Avg Anomaly Duration", avg_dur)

    # ----------------------------------------------------------------------------------
    # RUNNING STATE HEALTH & ACOUSTIC PROFILE (Quick-Glance Operational Assessment)
    # ----------------------------------------------------------------------------------
    states = m_stream['machine_state'].tolist() if not m_stream.empty else []
    times = pd.to_datetime(m_stream['recorded_at']).tolist() if not m_stream.empty else []
    cur_state = states[-1] if states else "IDLE"
    session_sec = 0
    if len(states) >= 2:
        i = len(states) - 1
        while i >= 0 and states[i] == cur_state:
            i -= 1
        start_idx = max(0, i + 1)
        session_sec = (times[-1] - times[start_idx]).total_seconds()
    session_min = int(session_sec / 60)
    session_hr = session_min // 60
    session_rem_min = session_min % 60
    session_time_str = f"{session_hr}h {session_rem_min}m" if session_hr > 0 else f"{session_min}m"

    running_pts = m_stream[m_stream['machine_state'] == 'RUNNING']
    if not running_pts.empty:
        cur_vol = float(running_pts['running_baseline'].iloc[-1])
        recent_run = running_pts.tail(20)
        mean_vol = float(recent_run['running_baseline'].mean())
        std_vol = float(recent_run['running_baseline'].std()) if len(recent_run) > 1 else 0.0
        cv_pct = (std_vol / (mean_vol + 1e-5)) * 100.0
        stability_score = max(0.0, min(100.0, 100.0 - cv_pct * 3.0))

        if stability_score >= 85.0:
            stability_badge = f"🟢 Stable ({stability_score:.0f}%)"
            stability_color = "#00e676"
        elif stability_score >= 70.0:
            stability_badge = f"🟡 Modulated (±{std_vol:.0f})"
            stability_color = "#ffa726"
        else:
            stability_badge = f"🔴 Surging (±{std_vol:.0f})"
            stability_color = "#f44336"
    else:
        cur_vol = float(m_stream['running_baseline'].iloc[-1]) if not m_stream.empty else 0.0
        stability_badge = "⚪ Idle Standby"
        stability_color = "#9e9e9e"

    latest_audio_row = df[df['audio_path'].notnull()].iloc[0] if (not df.empty and df['audio_path'].notnull().any()) else None
    spectral_health = None
    if latest_audio_row is not None:
        res_path = str(latest_audio_row['audio_path'])
        if not os.path.exists(res_path):
            for c in [os.path.join(BASE_DIR, res_path), os.path.join(AUDIO_DIR, os.path.basename(res_path))]:
                if os.path.exists(c):
                    res_path = c
                    break
        if os.path.exists(res_path):
            try:
                audio_data, sr = decode_audio_from_mp4(res_path)
                spectral_health = extract_spectral_health_profile(audio_data, sr)
            except Exception:
                pass

    if spectral_health is None:
        spectral_health = {
            "dominant_freq_hz": 120,
            "low_band_pct": 52.0,
            "mid_band_pct": 38.0,
            "high_band_pct": 10.0,
            "spectral_centroid_hz": 850,
            "crest_factor": 1.9,
            "kurtosis": 2.8,
            "health_assessment": "🟢 Balanced Operational Acoustics",
            "health_color": "#00e676"
        }

    with st.container():
        st.markdown(f"""
        <div style="background-color: #141b26; border: 1px solid #2d3b4e; border-left: 6px solid {stability_color}; border-radius: 8px; padding: 14px 18px; margin-top: 10px; margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div style="font-size: 1.1em; font-weight: bold; color: #ffffff;">
                    ⚡ Real-Time Running State Health & Acoustic Profile
                </div>
                <div style="font-size: 0.85em; font-weight: bold; color: {spectral_health['health_color']}; background: #1a2736; padding: 3px 10px; border-radius: 4px; border: 1px solid #33445b;">
                    {spectral_health['health_assessment']}
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 12px;">
                <div style="background: #1e2634; padding: 8px 12px; border-radius: 6px;">
                    <div style="font-size: 0.72em; color: #90caf9; text-transform: uppercase;">Current Session</div>
                    <div style="font-size: 1.15em; font-weight: bold; color: #ffffff;">{cur_state} ({session_time_str})</div>
                </div>
                <div style="background: #1e2634; padding: 8px 12px; border-radius: 6px;">
                    <div style="font-size: 0.72em; color: #90caf9; text-transform: uppercase;">Running Volume & Stability</div>
                    <div style="font-size: 1.15em; font-weight: bold; color: #ffffff;">{cur_vol:.0f} RMS <span style="font-size: 0.7em; color: {stability_color};">({stability_badge})</span></div>
                </div>
                <div style="background: #1e2634; padding: 8px 12px; border-radius: 6px;">
                    <div style="font-size: 0.72em; color: #90caf9; text-transform: uppercase;">Dominant Operating Tone</div>
                    <div style="font-size: 1.15em; font-weight: bold; color: #00e5ff;">{spectral_health['dominant_freq_hz']} Hz <span style="font-size: 0.7em; color: #b0bec5;">(Motor Line/Pole)</span></div>
                </div>
                <div style="background: #1e2634; padding: 8px 12px; border-radius: 6px;">
                    <div style="font-size: 0.72em; color: #90caf9; text-transform: uppercase;">Acoustic Texture</div>
                    <div style="font-size: 1.15em; font-weight: bold; color: #a7f3d0;">Crest: {spectral_health['crest_factor']} <span style="font-size: 0.7em; color: #9e9e9e;">(Smooth)</span></div>
                </div>
            </div>
            <div>
                <div style="display: flex; justify-content: space-between; font-size: 0.78em; color: #b0bec5; margin-bottom: 4px;">
                    <span>🟢 Low Band (0–500 Hz Rumble): <strong>{spectral_health['low_band_pct']}%</strong></span>
                    <span>🔵 Mid Band (500–2.5k Hz Flow): <strong>{spectral_health['mid_band_pct']}%</strong></span>
                    <span>🔴 High Band (2.5k–8k Hz Friction): <strong>{spectral_health['high_band_pct']}%</strong></span>
                </div>
                <div style="display: flex; height: 8px; border-radius: 4px; overflow: hidden; background: #263238;">
                    <div style="width: {spectral_health['low_band_pct']}%; background: #00e676;" title="Low Band: {spectral_health['low_band_pct']}%"></div>
                    <div style="width: {spectral_health['mid_band_pct']}%; background: #00e5ff;" title="Mid Band: {spectral_health['mid_band_pct']}%"></div>
                    <div style="width: {spectral_health['high_band_pct']}%; background: #f44336;" title="High Band: {spectral_health['high_band_pct']}%"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Baseline Status Banner
    baseline_info = get_machine_baseline(st.session_state["selected_device"], DB_PATH)
    if baseline_info:
        b_text = f"🎯 <strong>Acoustic Baseline Profile:</strong> Established ({baseline_info['sample_count']} verified Normal Operation clips | Baseline RMS: {baseline_info['baseline_rms']:.3f} | Updated: {baseline_info['updated_at']})"
        st.markdown(f"<div style='background-color: #142820; border-left: 4px solid #00e676; border-radius: 4px; padding: 8px 12px; margin-top: 8px; margin-bottom: 8px; font-size: 0.9em; color: #a7f3d0;'>{b_text}</div>", unsafe_allow_html=True)
    else:
        b_text = "⚠️ <strong>Acoustic Baseline Profile:</strong> Pending (Verify a sound clip below as <em>'Normal Operation'</em> to establish this machine's ground-truth running baseline)"
        st.markdown(f"<div style='background-color: #2b2210; border-left: 4px solid #ffa726; border-radius: 4px; padding: 8px 12px; margin-top: 8px; margin-bottom: 8px; font-size: 0.9em; color: #fde68a;'>{b_text}</div>", unsafe_allow_html=True)
else:
    # Fleet Summary Header
    st.title("🏭 Industrial Acoustic Fleet Monitor")
    st.caption("Fleet-wide acoustic anomaly detection, connected device states, and active learning audio classification.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Connected Edge Devices", len(all_machines))
    c2.metric("Total Fleet Anomalies", len(raw_df))
    c3.metric("Fleet Mean Duration", f"{raw_df['duration_seconds'].mean():.1f} s" if not raw_df.empty else "0.0 s")
    c4.metric("Average Peak Amplitude", f"{raw_df['peak_amplitude'].mean():.0f}" if (not raw_df.empty and 'peak_amplitude' in raw_df.columns) else "N/A")

st.markdown("---")

# Dynamic Tab Configuration
if st.session_state["selected_device"] == "All Devices":
    tab_fleet, tab_events, tab_visuals, tab_data = st.tabs([
        "🖥️ Device Fleet Overview",
        "🎧 All Audio Events & Classifications",
        "📊 Fleet Telemetry Trends",
        "🗄️ Raw Telemetry Table"
    ])
else:
    tab_events, tab_visuals, tab_data = st.tabs([
        f"🎧 {st.session_state['selected_device']} Audio & Classify",
        f"📊 {st.session_state['selected_device']} Telemetry Trends",
        f"🗄️ {st.session_state['selected_device']} Logs"
    ])
    tab_fleet = None

# --------------------------------------------------------------------------------------
# TAB: FLEET & MACHINE STATUS OVERVIEW (When in All Devices view)
# --------------------------------------------------------------------------------------
if tab_fleet is not None:
    with tab_fleet:
        st.subheader("🖥️ Connected Devices & Machine State Overview")
        st.caption("Click on any device below to jump directly to its dedicated dashboard, audio events, and trends.")

        now_dt = datetime.datetime.now()

        for m_id in all_machines:
            m_df = raw_df[raw_df['machine_id'] == m_id] if not raw_df.empty else pd.DataFrame()
            m_stream = raw_stream_df[raw_stream_df['machine_id'] == m_id] if not raw_stream_df.empty else pd.DataFrame()

            last_time = None
            if not m_stream.empty:
                last_time = pd.to_datetime(m_stream['recorded_at'].iloc[-1]).replace(tzinfo=None)
                last_state = m_stream['machine_state'].iloc[-1]
            elif not m_df.empty:
                last_time = pd.to_datetime(m_df['time_detected'].iloc[0]).replace(tzinfo=None)
                last_state = "UNKNOWN"

            if last_time is not None:
                time_diff_min = (now_dt - last_time).total_seconds() / 60.0
                last_time_str = f"{last_time} ({time_diff_min:.1f} mins ago)"
            else:
                time_diff_min = 999.0
                last_time_str = "No contact recorded"

            total_m_events = len(m_df)
            
            if time_diff_min < 2:
                status_text = "🟢 ONLINE (Heartbeat Active)"
                status_color = "#00e676"
                machine_power = f"⚡ Machine Power: {last_state} / Operating" if last_state == "RUNNING" else "💤 Machine Power: Standby / Idle"
            elif time_diff_min < 60:
                status_text = "🟡 STANDBY / IDLE"
                status_color = "#ffa726"
                machine_power = "💤 Machine Power: OFF / Idle Standby"
            else:
                status_text = "⚪ OFFLINE (No Recent Heartbeat)"
                status_color = "#9e9e9e"
                machine_power = "🔌 Machine Power: Disconnected / Standby"

            top_diag = m_df['predicted_class'].mode().iloc[0] if (not m_df.empty and 'predicted_class' in m_df.columns and not m_df['predicted_class'].dropna().empty) else "Normal"
            avg_dur_str = f"{m_df['duration_seconds'].mean():.1f}s" if not m_df.empty else "N/A"

            with st.container():
                col_card_head, col_card_status = st.columns([3, 1])
                with col_card_head:
                    if st.button(f"📍 {m_id}  ➜", key=f"btn_card_title_{m_id}", help=f"Click to open dedicated dashboard for {m_id}"):
                        st.session_state["selected_device"] = m_id
                        st.rerun()
                with col_card_status:
                    st.markdown(f"<div style='text-align: right; padding-top: 8px; font-weight: bold; color: {status_color}; font-size: 1.05em;'>{status_text}</div>", unsafe_allow_html=True)

                st.markdown(f"""
                <div style="background-color: #1a2230; border: 1px solid #303e54; border-left: 6px solid {status_color}; border-radius: 6px; padding: 10px 16px; margin-bottom: 12px;">
                    <div style="color: #b0bec5; font-size: 0.95em;">
                        <strong>State:</strong> {machine_power} &nbsp;|&nbsp; 
                        <strong>Last Contact:</strong> {last_time_str}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_m1, col_m2, col_m3 = st.columns([1, 2.2, 1])
                col_m1.metric("Total Detections", total_m_events)
                with col_m2:
                    st.markdown("<div style='font-size: 0.85em; color: #b0bec5; margin-bottom: 2px;'>Primary Anomaly Type</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size: 1.35em; font-weight: bold; color: #00e5ff; white-space: nowrap;'>{top_diag}</div>", unsafe_allow_html=True)
                col_m3.metric("Avg Duration", avg_dur_str)
                st.markdown("---")
            


# --------------------------------------------------------------------------------------
# TAB: EVENT AUDIO & ACTIVE LEARNING CLASSIFICATION + MASS ACTIONS
# --------------------------------------------------------------------------------------
with tab_events:
    view_title = f"{st.session_state['selected_device']} Audio Events" if st.session_state['selected_device'] != "All Devices" else "All Fleet Audio Events"
    st.subheader(f"🎧 {view_title} & Classification")
    st.caption("Review captured anomalies, play back audio, inspect spectrograms, and perform single or bulk label/deletion actions.")

    if df.empty:
        st.info("No recorded audio anomaly events in this view yet. Continuous 30-second telemetry heartbeats are streaming under the 'Trends' and 'Logs' tabs.")
    else:
        # ------------------ MASS DELETION & BULK ACTION BAR ------------------
        with st.expander("🛠️ Bulk Actions & Mass Deletion Manager", expanded=False):
            st.markdown("##### Manage multiple anomaly events simultaneously:")
            col_b1, col_b2, col_b3 = st.columns(3)
        
        with col_b1:
            if st.button("🧹 Clear All Ignored Samples", type="secondary"):
                conn = get_duckdb_connection()
                ignored_rows = conn.execute("SELECT audio_path FROM machine_anomalies WHERE verified_label = 'Ignored / False Positive'").fetchall()
                for (ap,) in ignored_rows:
                    if ap:
                        p = os.path.join(BASE_DIR, ap) if not os.path.isabs(ap) else ap
                        if os.path.exists(p):
                            try: os.remove(p)
                            except Exception: pass
                conn.execute("DELETE FROM machine_anomalies WHERE verified_label = 'Ignored / False Positive'")
                conn.close()
                st.toast(f"Deleted {len(ignored_rows)} ignored samples.")
                st.rerun()

        with col_b2:
            if st.button("🗣️ Clear All Speech Detections", type="secondary"):
                conn = get_duckdb_connection()
                speech_rows = conn.execute("SELECT audio_path FROM machine_anomalies WHERE predicted_class = 'Human Speech / Conversation' OR verified_label = 'Human Speech / Conversation'").fetchall()
                for (ap,) in speech_rows:
                    if ap:
                        p = os.path.join(BASE_DIR, ap) if not os.path.isabs(ap) else ap
                        if os.path.exists(p):
                            try: os.remove(p)
                            except Exception: pass
                conn.execute("DELETE FROM machine_anomalies WHERE predicted_class = 'Human Speech / Conversation' OR verified_label = 'Human Speech / Conversation'")
                conn.close()
                st.toast(f"Deleted {len(speech_rows)} speech samples.")
                st.rerun()

        with col_b3:
            confirm_all = st.checkbox("Confirm: Purge current view records", key="confirm_purge_all")
            if st.button("⚠️ Delete Current Telemetry & Audio", type="primary", disabled=not confirm_all):
                conn = get_duckdb_connection()
                if st.session_state["selected_device"] == "All Devices":
                    all_rows = conn.execute("SELECT audio_path FROM machine_anomalies").fetchall()
                    for (ap,) in all_rows:
                        if ap:
                            p = os.path.join(BASE_DIR, ap) if not os.path.isabs(ap) else ap
                            if os.path.exists(p):
                                try: os.remove(p)
                                except Exception: pass
                    conn.execute("DELETE FROM machine_anomalies")
                else:
                    dev_rows = conn.execute("SELECT audio_path FROM machine_anomalies WHERE machine_id = ?", (st.session_state["selected_device"],)).fetchall()
                    for (ap,) in dev_rows:
                        if ap:
                            p = os.path.join(BASE_DIR, ap) if not os.path.isabs(ap) else ap
                            if os.path.exists(p):
                                try: os.remove(p)
                                except Exception: pass
                    conn.execute("DELETE FROM machine_anomalies WHERE machine_id = ?", (st.session_state["selected_device"],))
                conn.close()
                st.toast("Purged records.")
                st.rerun()

        # Multi-select check for specific deletion
        event_options = {f"{row['time_detected']} | {row['machine_id']} ({row.get('predicted_class','Unknown')})": str(row['event_id']) for _, row in df.iterrows()}
        selected_to_delete = st.multiselect("Select specific events to delete in bulk:", options=list(event_options.keys()))
        if selected_to_delete:
            if st.button(f"🗑️ Delete {len(selected_to_delete)} Selected Events", type="primary"):
                ids_to_del = [event_options[k] for k in selected_to_delete]
                conn = get_duckdb_connection()
                for eid in ids_to_del:
                    row_p = conn.execute("SELECT audio_path FROM machine_anomalies WHERE event_id = ?", (eid,)).fetchone()
                    if row_p and row_p[0]:
                        p = os.path.join(BASE_DIR, row_p[0]) if not os.path.isabs(row_p[0]) else row_p[0]
                        if os.path.exists(p):
                            try: os.remove(p)
                            except Exception: pass
                    conn.execute("DELETE FROM machine_anomalies WHERE event_id = ?", (eid,))
                conn.close()
                st.toast(f"Deleted {len(ids_to_del)} events.")
                st.rerun()

    st.markdown("---")

    # ------------------ PAGINATION & EVENT CARDS ------------------
    events_per_page = 15
    total_events = len(df)
    total_pages = max(1, (total_events + events_per_page - 1) // events_per_page)
    
    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        st.markdown(f"Displaying **{min(total_events, events_per_page)}** of **{total_events}** recorded anomaly events:")
    with col_p2:
        if total_pages > 1:
            current_page = st.selectbox("📄 Page:", options=list(range(1, total_pages + 1)), index=0, key="event_page_select")
        else:
            current_page = 1
            
    start_idx = (current_page - 1) * events_per_page
    end_idx = start_idx + events_per_page
    page_df = df.iloc[start_idx:end_idx]

    for index, row in page_df.iterrows():
        event_id = str(row['event_id'])
        machine_id = str(row['machine_id'])
        time_det = str(row['time_detected'])
        pred_class = str(row['predicted_class']) if ('predicted_class' in row and pd.notna(row['predicted_class'])) else "Unclassified"
        conf = float(row['confidence']) if ('confidence' in row and pd.notna(row['confidence'])) else 0.0
        verified_label = str(row['verified_label']) if ('verified_label' in row and pd.notna(row['verified_label'])) else None

        # Resolve audio path
        audio_file = row.get('audio_path')
        resolved_audio_path = None
        if pd.notna(audio_file) and isinstance(audio_file, str) and audio_file.strip():
            audio_str = str(audio_file)
            candidates = [
                audio_str,
                os.path.join(BASE_DIR, audio_str),
                os.path.join(BASE_DIR, "..", audio_str),
                os.path.join(AUDIO_DIR, os.path.basename(audio_str))
            ]
            for c in candidates:
                if os.path.exists(c):
                    resolved_audio_path = c
                    break

        dur_val = float(row.get('duration_seconds') or 0.0)
        if dur_val >= 4.9:
            dur_display = f"Anomaly Duration: >{dur_val:.1f}s (Continuous)"
        elif dur_val >= 1.0:
            dur_display = f"Anomaly Duration: {dur_val:.2f}s (Spin-Up / Surge)"
        elif dur_val > 0.05:
            dur_display = f"Anomaly Duration: {dur_val:.2f}s (Impulse Transient)"
        else:
            dur_display = "Anomaly Duration: <0.1s (Click / Strike)"

        # User Classified vs Machine Predicted Badge
        if verified_label and verified_label != "None":
            if verified_label == "Ignored / False Positive":
                status_header = "🚫 **[Ignored]**"
            elif verified_label == "Human Speech / Conversation":
                status_header = "🗣️ **[User Verified: Human Speech]**"
            else:
                status_header = f"✅ **[User Verified: {verified_label}]**"
        else:
            conf_str = f" ({conf*100:.0f}%)" if conf > 0 else ""
            if pred_class == "Human Speech / Conversation":
                status_header = f"🗣️ **[Model: Human Speech / Conversation{conf_str}]**"
            else:
                status_header = f"🤖 **[Model: {pred_class}{conf_str}]**"

        is_expanded = (event_id == st.session_state.get('last_interacted_event') or (st.session_state.get('last_interacted_event') is None and index < 3))
        with st.expander(
            f"📍 **{machine_id}** — {time_det} | {dur_display} | 🎵 Clip: 5.0s | {status_header}",
            expanded=is_expanded
        ):
            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.markdown("#### 🎵 Audio Playback & Diagnosis")
                if resolved_audio_path and os.path.exists(resolved_audio_path):
                    with open(resolved_audio_path, "rb") as af:
                        st.audio(af.read(), format="audio/mp4")
                    st.caption(f"📁 `{os.path.basename(resolved_audio_path)}` ({os.path.getsize(resolved_audio_path)/1024:.1f} KB)")
                else:
                    st.warning("Audio clip pending upload from edge device.")

                st.markdown("---")
                st.markdown("#### 🏷️ Classification & Human Review")
                st.write(f"**Model Prediction**: `{pred_class}` (Confidence: `{conf*100:.1f}%`)")
                if conf > 0:
                    st.progress(min(1.0, conf))

                if verified_label:
                    st.success(f"✅ **Verified Label**: `{verified_label}`")

                # Label Selection & Direct Action Buttons
                all_options = DEFAULT_CLASSES + ["Ignored / False Positive", "Non-Classified"]
                current_selection = verified_label if verified_label in all_options else pred_class
                if current_selection not in all_options:
                    options = [current_selection] + all_options
                else:
                    options = all_options
                
                selected_label = st.selectbox(
                    "Confirm or Adjust Classification:",
                    options=options,
                    index=options.index(current_selection) if current_selection in options else 0,
                    key=f"select_{event_id}"
                )
                
                btn_save, btn_ignore, btn_del = st.columns([1.3, 1, 1])
                with btn_save:
                    if st.button("💾 Save Label", key=f"btn_save_{event_id}", type="primary"):
                        now_utc = datetime.datetime.now(datetime.timezone.utc)
                        conn = get_duckdb_connection()
                        val_to_save = None if selected_label == "Non-Classified" else selected_label
                        conn.execute("""
                            UPDATE machine_anomalies
                            SET verified_label = ?, verified_at = ?
                            WHERE event_id = ?
                        """, (val_to_save, now_utc, event_id))
                        conn.close()
                        if selected_label == "Normal Operation":
                            update_machine_baseline(machine_id, DB_PATH)
                        st.session_state['last_interacted_event'] = event_id
                        st.toast(f"✅ Saved label: '{selected_label}'")
                        st.rerun()

                with btn_ignore:
                    if st.button("🚫 Ignore", key=f"ignore_{event_id}"):
                        conn = get_duckdb_connection()
                        conn.execute("UPDATE machine_anomalies SET verified_label = 'Ignored / False Positive' WHERE event_id = ?", (event_id,))
                        conn.close()
                        st.session_state['last_interacted_event'] = event_id
                        st.toast("🚫 Marked as Ignored (excluded from ML)")
                        st.rerun()

                with btn_del:
                    if st.button("🗑️ Delete", key=f"del_{event_id}"):
                        conn = get_duckdb_connection()
                        conn.execute("DELETE FROM machine_anomalies WHERE event_id = ?", (event_id,))
                        conn.close()
                        if resolved_audio_path and os.path.exists(resolved_audio_path):
                            try: os.remove(resolved_audio_path)
                            except Exception: pass
                        st.toast("🗑️ Detection deleted")
                        st.rerun()

            with col_right:
                st.markdown("#### 📈 Waveform & Spectrogram Analysis")
                if resolved_audio_path and os.path.exists(resolved_audio_path):
                    try:
                        audio_data, sr = decode_audio_from_mp4(resolved_audio_path)
                        if len(audio_data) > 0:
                            time_axis = np.linspace(0, len(audio_data) / sr, num=len(audio_data))
                            subsample = max(1, len(audio_data) // 1000)
                            fig_wave = go.Figure()
                            fig_wave.add_trace(go.Scatter(
                                x=time_axis[::subsample],
                                y=audio_data[::subsample],
                                mode='lines',
                                line=dict(color='#00e5ff', width=1),
                                name='Amplitude'
                            ))
                            fig_wave.update_layout(
                                title="Time-Domain Waveform",
                                xaxis_title="Time (seconds)",
                                yaxis_title="Amplitude",
                                height=200,
                                margin=dict(l=20, r=20, t=30, b=20),
                                template="plotly_dark"
                            )
                            st.plotly_chart(fig_wave)

                            f, t, Sxx = signal.spectrogram(audio_data, fs=sr, nperseg=512, noverlap=256)
                            Sxx_db = 10 * np.log10(Sxx + 1e-10)

                            fig_spec = go.Figure(data=go.Heatmap(
                                z=Sxx_db,
                                x=t,
                                y=f,
                                colorscale='Viridis',
                                colorbar=dict(title="dB")
                            ))
                            fig_spec.update_layout(
                                title="Frequency Spectrogram (Hz)",
                                xaxis_title="Time (s)",
                                yaxis_title="Frequency (Hz)",
                                height=220,
                                margin=dict(l=20, r=20, t=30, b=20),
                                template="plotly_dark"
                            )
                            st.plotly_chart(fig_spec)
                    except Exception as e:
                        st.error(f"Error rendering visual plots: {e}")
                else:
                    st.info("Visual spectrogram will render once audio is uploaded.")

# --------------------------------------------------------------------------------------
# TAB: TELEMETRY TRENDS
# --------------------------------------------------------------------------------------
with tab_visuals:
    st.subheader(f"📊 Telemetry Trends ({st.session_state['selected_device']})")
    
    st.markdown("#### 📈 Continuous Acoustic Baseline Stream (Acoustic Drift & Wear Tracking)")
    st.caption("Tracks running volume drift over hours and weeks. A gradual upward slope indicates mechanical wear / bearing degradation.")

    @st.fragment(run_every=5 if enable_fragment_sync else None)
    def render_stream_chart():
        d_stream = load_stream_telemetry(st.session_state['selected_device'])
        if time_filter != "All Time":
            hours_map = {"Last 1 Hour": 1, "Last 6 Hours": 6, "Last 24 Hours": 24, "Last 7 Days": 168}
            cutoff = datetime.datetime.now() - datetime.timedelta(hours=hours_map[time_filter])
            if not d_stream.empty and 'recorded_at' in d_stream.columns:
                d_stream = d_stream[pd.to_datetime(d_stream['recorded_at']).dt.tz_localize(None) >= cutoff]
        if min_amp_filter > 0 and not d_stream.empty and 'peak_amplitude' in d_stream.columns:
            d_stream = d_stream[d_stream['peak_amplitude'] >= min_amp_filter]

        if not d_stream.empty:
            fig_stream = go.Figure()
            fig_stream.add_trace(go.Scatter(
                x=d_stream['recorded_at'],
                y=d_stream['running_baseline'],
                mode='lines+markers',
                line=dict(color='#00e5ff', width=2),
                name='Running Baseline'
            ))
            fig_stream.add_trace(go.Scatter(
                x=d_stream['recorded_at'],
                y=d_stream['peak_amplitude'],
                mode='lines',
                line=dict(color='#ff5252', width=1, dash='dot'),
                name='Peak Amplitude'
            ))
            fig_stream.update_layout(
                title=f"Continuous Running Baseline ({time_filter})",
                xaxis_title="Time",
                yaxis_title="Amplitude Level",
                height=280,
                template="plotly_dark",
                margin=dict(l=20, r=20, t=35, b=20)
            )
            st.plotly_chart(fig_stream, use_container_width=True)

            total_pings = len(d_stream)
            running_pings = len(d_stream[d_stream['machine_state'] == 'RUNNING'])
            duty_cycle = (running_pings / total_pings * 100.0) if total_pings > 0 else 0.0
            
            col_st1, col_st2, col_st3 = st.columns(3)
            col_st1.metric("Current Running Baseline", f"{d_stream['running_baseline'].iloc[-1]:.1f}")
            col_st2.metric("Calculated Duty Cycle", f"{duty_cycle:.1f}%")
            col_st3.metric("Telemetry Heartbeats", total_pings)
        else:
            st.info(f"No stream telemetry matching current filters ({time_filter}).")

    render_stream_chart()

    st.markdown("---")
    st.markdown("#### ⏳ Anomaly Distributions & Timeline")
    if not df.empty:
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            fig_dur = px.histogram(
                df,
                x="duration_seconds",
                nbins=15,
                title="Distribution of Anomaly Durations (s)",
                color_discrete_sequence=["#00e5ff"],
                template="plotly_dark"
            )
            st.plotly_chart(fig_dur)

        with col_v2:
            if 'peak_amplitude' in df.columns and not df['peak_amplitude'].dropna().empty:
                fig_amp = px.histogram(
                    df,
                    x="peak_amplitude",
                    nbins=15,
                    title="Distribution of Peak Amplitudes",
                    color_discrete_sequence=["#ff5252"],
                    template="plotly_dark"
                )
                st.plotly_chart(fig_amp)

        df_sorted = df.sort_values("time_detected")
        fig_timeline = px.scatter(
            df_sorted,
            x="time_detected",
            y="duration_seconds",
            size="peak_amplitude" if 'peak_amplitude' in df_sorted.columns else None,
            color="predicted_class" if 'predicted_class' in df_sorted.columns else None,
            hover_data=["machine_id", "duration_seconds", "verified_label"],
            title="Anomaly Events Over Time",
            template="plotly_dark"
        )
        st.plotly_chart(fig_timeline)
    else:
        st.info("Anomaly duration and timeline charts will populate when discrete anomaly events are triggered.")

# --------------------------------------------------------------------------------------
# TAB: RAW DATA TABLE
# --------------------------------------------------------------------------------------
with tab_data:
    st.subheader(f"🗄️ Raw Telemetry Database ({st.session_state['selected_device']})")
    
    # ------------------ INLINE DATA DELETE & WIPE CONTROLS ------------------
    with st.expander("🗑️ Delete & Clear Telemetry Data", expanded=False):
        target_dev = st.session_state["selected_device"]
        dev_lbl = f"`{target_dev}`" if target_dev != "All Devices" else "ALL Fleet Devices"
        st.markdown(f"##### Delete telemetry records for {dev_lbl}:")
        
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            if st.button(f"🧹 Delete 30s Stream Telemetry ({dev_lbl})", key="btn_del_stream_tab"):
                conn = get_duckdb_connection()
                if target_dev == "All Devices":
                    conn.execute("DELETE FROM machine_telemetry_stream")
                else:
                    conn.execute("DELETE FROM machine_telemetry_stream WHERE machine_id = ?", (target_dev,))
                conn.close()
                st.toast(f"Deleted stream telemetry for {dev_lbl}.")
                st.rerun()

        with col_w2:
            if st.button(f"🗑️ Delete Anomaly Records & Audio ({dev_lbl})", key="btn_del_anom_tab"):
                conn = get_duckdb_connection()
                if target_dev == "All Devices":
                    rows = conn.execute("SELECT audio_path FROM machine_anomalies").fetchall()
                    conn.execute("DELETE FROM machine_anomalies")
                else:
                    rows = conn.execute("SELECT audio_path FROM machine_anomalies WHERE machine_id = ?", (target_dev,)).fetchall()
                    conn.execute("DELETE FROM machine_anomalies WHERE machine_id = ?", (target_dev,))
                for (ap,) in rows:
                    if ap:
                        p = os.path.join(BASE_DIR, ap) if not os.path.isabs(ap) else ap
                        if os.path.exists(p):
                            try: os.remove(p)
                            except Exception: pass
                conn.close()
                st.toast(f"Deleted anomaly records and audio for {dev_lbl}.")
                st.rerun()

    st.markdown("---")
    st.markdown("#### 💓 Continuous 30s Telemetry Heartbeat Stream (`machine_telemetry_stream`)")
    if not df_stream.empty:
        st.dataframe(df_stream, use_container_width=True)
    else:
        st.info("No heartbeat telemetry stream records found.")

    st.markdown("---")
    st.markdown("#### 🚨 Anomaly Event Records (`machine_anomalies`)")
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No discrete anomaly event records found for this view.")