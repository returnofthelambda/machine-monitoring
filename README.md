# 🏭 Industrial Acoustic Anomaly Detector & Fleet Management

An end-to-end industrial IoT condition monitoring platform powered by **Realtek AmebaPro2 (AMB82-MINI)** edge nodes, an event-driven **MQTT & HTTP pipeline**, an **embedded DuckDB analytics store**, and an **interactive Streamlit active-learning dashboard**.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                   AMB82-MINI Edge Sensor Nodes                         │
│                                                                        │
│   • 16kHz Continuous Monitoring (ARM Cortex-M33 DSP)                   │
│   • 5-Minute Store-and-Forward Wi-Fi Duty-Cycling (5.8x Battery Life)  │
│   • Self-Calibrating Ambient Noise Floor & Hysteresis State Machine    │
│   • On-Device Harmonic Voice Filter & Event-Driven MP4 Audio Capture   │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │ (MQTT Telemetry / Batch :1883) │ (HTTP Base64 POST :5000)
                    ▼                                ▼
┌──────────────────────────────┐    ┌────────────────────────────────────┐
│      MQTT Broker Node        │    │    Backend Server (Linux / VM)     │
│      (172.16.40.129)         │    │          (172.16.40.130)           │
│                              │    │                                    │
│ • Mosquitto Broker           │───▶│ • anomaly-ingest.service (MQTT)    │
│ • factory/+/telemetry/batch  │    │ • anomaly-upload.service (:5000)   │
│ • factory/+/anomalies        │    │ • anomaly-dashboard.service (:8501)│
│ • factory/+/cmd/ota          │    │ • machine_telemetry.duckdb         │
└──────────────────────────────┘    └────────────────────────────────────┘
                                                     │
                                                     ▼
                                    ┌────────────────────────────────────┐
                                    │    Interactive Streamlit UI        │
                                    │      http://<SERVER_IP>:8501       │
                                    │                                    │
                                    │ • Fleet Status & Duty Tracking     │
                                    │ • Paginated Event Cards & Audio    │
                                    │ • Active Learning ML Retraining    │
                                    │ • Remote One-Click OTA Firmware Push│
                                    └────────────────────────────────────┘
```

---

## 🚀 Quick Start: Edge Firmware Deployment

### 1. Configure `.env`
Edit [.env](file:///.env) before flashing each unit:
```ini
WIFI_SSID="factory_iot_wifi"
WIFI_PASSWORD="Your_Secure_Password"
MQTT_HOST="172.16.40.129"
MQTT_PORT=1883
MQTT_USER="factory_worker"
MQTT_PASSWORD="Your_Password"
MACHINE_ID="MACH_PUMP_01"
UPLOAD_HOST="172.16.40.130"
UPLOAD_PORT=5000
BOARD_FQBN="realtek:AmebaPro2:Ameba_AMB82-MINI:LoadNNModelSource=sd,OTAMode=Enable"
PORT="/dev/ttyUSB0"
```

### 2. Initial USB Flash (One-Time Setup)
1. Connect the AMB82-MINI via USB.
2. Enter **Download Mode**: Hold `DOWNLOAD`, press & release `RESET`, then release `DOWNLOAD`.
3. Compile and flash:
   ```bash
   ./build.sh --flash
   ```
4. Press `RESET`. The board connects to Wi-Fi and enables OTA support.

### 3. Over-The-Air (OTA) Remote Firmware Updates
After the initial USB flash, update nodes wirelessly without physical access:

1. **Build & Stage:**
   ```bash
   ./build.sh --deploy-ota
   ```
2. **Push Remotely:** Open the Streamlit dashboard sidebar under **`📡 Over-The-Air (OTA) Updates`** and click **`🚀 Push OTA Update`**, or publish over MQTT:
   ```bash
   mosquitto_pub -h 172.16.40.129 -p 1883 -u factory_worker -P Your_Password \
     -t "factory/MACH_PUMP_01/cmd/ota" \
     -m '{"resource":"ota/firmware.bin","port":5000}'
   ```

---

## 🖥️ Server Deployment (Debian / Ubuntu / Proxmox)

On the backend server (`172.16.40.130`):

### 1. Install Dependencies & Setup Directory
```bash
sudo apt update && sudo apt install -y python3-venv python3-pip libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev
sudo mkdir -p /opt/anomaly-av && sudo chown -R $USER:$USER /opt/anomaly-av
cd /opt/anomaly-av && python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

### 2. Enable Systemd Services
Deploy three auto-restarting services:
* **`anomaly-upload.service`**: Receives audio uploads, extracts features, serves OTA binaries on port `5000`.
* **`anomaly-ingest.service`**: Ingests real-time and 5-minute batched telemetry into DuckDB.
* **`anomaly-dashboard.service`**: Hosts the Streamlit UI on port `8501`.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now anomaly-upload anomaly-ingest anomaly-dashboard
```

---

## ⚡ Key System Features

* **🔋 5-Minute Store-and-Forward Wi-Fi Duty-Cycling:** Disables the Wi-Fi radio for 4m 58s out of every 5 minutes ($I_{\text{avg}} = 31.0\text{ mA}$), extending battery life by **$5.8\times$** while maintaining continuous 30-second trend granularity in DuckDB.
* **🚨 Instant Anomaly & Leading-Edge Capture:** Startup inrush ($<20\text{ms}$ contactor snap), shutdown spin-down, or loud in-run spikes immediately wake Wi-Fi, upload audio, and publish alerts in $<2\text{s}$.
* **⏱️ True Acoustic Duration Measurement:** Evaluates a 25ms sliding RMS energy envelope over the background noise floor to measure exact transient durations (e.g. `0.15s` contactor snap, `0.68s` startup surge, `1.60s` spin-up) rather than hardcoded edge guesses.
* **⚡ Real-Time Running State Health & Acoustic Profiling:** Live amplitude stability index ($CV$ jitter: `🟢 Stable` vs `🔴 Surging`), dominant operating tone (e.g. `120 Hz` motor fundamental), and a 3-band spectral balance bar (🟢 Low 0–500Hz, 🔵 Mid 500–2.5kHz, 🔴 High 2.5k–8kHz).
* **⚙️ Operational State Machine with Hysteresis:** Latching state machine (`IDLE` $\leftrightarrow$ `RUNNING`) eliminates runaway retrigger loops and tracks ongoing session run-time (e.g. `Running for 38m` / `Idle for 1h 12m`).
* **🧠 Active Learning ML Retraining:** Label events in the dashboard, compute ground-truth baseline vectors, and train a Random Forest model with automated historical reclassification.
* **📡 Dedicated One-Click OTA Updates:** Staged firmware binaries can be pushed wirelessly to individual machines or the entire fleet directly from the dashboard sidebar.
* **📄 Paginated High-Performance Dashboard:** Slices event cards (15 per page) for sub-50ms render times and non-blocking `@st.fragment` background synchronization.

---

## 📚 Detailed Technical Documentation

* 🔬 **[Hardware Selection & Low-Power Silicon Alternatives](docs/hardware_and_power.md)** — AMB82-MINI PoC role, comparison matrix with Ambiq Apollo4, Syntiant NDP120, Nordic nRF5340, STM32U5, and duty-cycling electrical math.
* 📦 **[Enclosures, Environmental Sealing & PCB Hardening](docs/enclosures_and_hardening.md)** — 3D-printable cases, IP65/IP67 junction boxes, GORE acoustic vents, SoC dust resilience, and conformal coating.
* 🧠 **[Active Learning & Acoustic Signal Processing Engine](docs/ml_and_audio_pipeline.md)** — Acoustic physics feature extraction, on-device pitch autocorrelation, dual-tier self-calibrating EMA engine, and supervised retraining.
* 🗺️ **[Multi-Modal Audio-Visual & Edge Vision Roadmap](docs/roadmap_vision.md)** — Synchronized FHD H.264 video capture and on-chip NPU YOLOv7-Tiny deployment plans.
