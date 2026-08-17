# 🏭 Industrial Acoustic Anomaly Detector & Fleet Management

An end-to-end industrial IoT monitoring system powered by the **Realtek AmebaPro2 (AMB82-MINI)** edge AI board, an event-driven **MQTT & HTTP ingestion pipeline**, an **embedded DuckDB analytics store**, and an **interactive Streamlit active-learning dashboard**.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                   AMB82-MINI Edge Sensor Nodes                         │
│                                                                        │
│   • 16kHz Continuous Acoustic Monitoring (ARM Cortex-M33 DSP)          │
│   • Dynamic 30s Telemetry Heartbeats (EMA Baseline Tracking)           │
│   • Real-Time On-Device Voice / Speech Harmonic Filter                 │
│   • Event-Driven 5.0s AAC/MP4 Hardware Audio Capture                   │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │ (MQTT Telemetry Alerts :1883)  │ (HTTP Base64 POST :5000)
                    ▼                                ▼
┌──────────────────────────────┐    ┌────────────────────────────────────┐
│      MQTT Broker Node        │    │    Backend Server (Linux / VM)     │
│      (192.168.1.50)          │    │          (192.168.1.100)           │
│                              │    │                                    │
│ • Mosquitto MQTT Broker      │───▶│ • anomaly-ingest.service (MQTT)    │
│ • factory/+/anomalies        │    │ • anomaly-upload.service (:5000)   │
│                              │    │ • anomaly-dashboard.service (:8501)│
│                              │    │ • machine_telemetry.duckdb         │
└──────────────────────────────┘    └────────────────────────────────────┘
                                                     │
                                                     ▼
                                    ┌────────────────────────────────────┐
                                    │    Interactive Streamlit UI        │
                                    │      http://<SERVER_IP>:8501       │
                                    │                                    │
                                    │ • Fleet & Machine State Overview   │
                                    │ • Audio Player & Spectrogram Heatmap│
                                    │ • Human-in-the-Loop Active Retrain │
                                    │ • Bulk Management & Mass Deletion  │
                                    └────────────────────────────────────┘
```

---

## 🚀 Setting Up & Deploying Multiple AMB82-MINI Devices

Deploying a fleet of AMB82-MINI sensors across a facility is fully automated via parameterized build flags.

### 1. Network & DHCP Configuration
1. Note the MAC address printed on the AMB82-MINI module or displayed during initial boot.
2. In your router (e.g., OpenWrt dashboard at `Network` -> `DHCP and DNS` -> `Static Leases`), create a static IP reservation for each board:
   - Device 1: `192.168.1.101` → `MACH_PUMP_01` (Water Feed Pump 1)
   - Device 2: `192.168.1.102` → `MACH_PUMP_02` (Cooling Loop Pump 2)
   - Device 3: `192.168.1.103` → `MACH_COMPRESSOR_01` (Air Compressor 1)
3. Ensure the Wi-Fi AP is configured for **2.4 GHz WPA2-PSK (`psk2`)**.

### 2. Configure `.env` for Each Target Device
Edit `.env` before flashing each unit:

```ini
# WiFi Configuration
WIFI_SSID="factory_iot_wifi"
WIFI_PASSWORD="Your_Secure_Password"

# MQTT Broker Configuration
MQTT_HOST="192.168.1.50"
MQTT_PORT=1883
MQTT_USER="factory_worker"
MQTT_PASSWORD="Your_MQTT_Password"

# Machine Identity (Change per device!)
MACHINE_ID="MACH_PUMP_02"
MQTT_TOPIC="factory/machine_02/anomalies"

# Backend Upload Endpoint
UPLOAD_HOST="192.168.1.100"
UPLOAD_PORT=5000

# Flashing Settings
PORT="/dev/ttyUSB0"
BOARD_FQBN="realtek:AmebaPro2:Ameba_AMB82-MINI:LoadNNModelSource=sd,OTAMode=Enable"
```

### 3. Flash the Device
1. Connect the AMB82-MINI via USB.
2. Enter **Download Mode**: Hold `DOWNLOAD`, press & release `RESET`, then release `DOWNLOAD`.
3. Compile and flash:
   ```bash
   ./build.sh --flash
   ```
4. Press `RESET` once finished. Repeat for each additional sensor node.

---

## 🖥️ Server Deployment (Debian / Ubuntu / Proxmox)

On the backend server (`192.168.1.100`):

### 1. Install System Dependencies
```bash
sudo apt update && sudo apt install -y python3-venv python3-pip libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev
```

### 2. Set Up Virtual Environment & Dependencies
```bash
sudo mkdir -p /opt/anomaly-av
sudo chown -R $USER:$USER /opt/anomaly-av
cd /opt/anomaly-av
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

### 3. Systemd Services Setup
Deploy three auto-restarting systemd units:

- **`/etc/systemd/system/anomaly-upload.service`**: Receives Base64 `.mp4` uploads, runs feature extraction, and performs initial ML classification.
- **`/etc/systemd/system/anomaly-ingest.service`**: Subscribes to `factory/+/anomalies` and `factory/+/telemetry` on the Mosquitto broker and records data into DuckDB.
- **`/etc/systemd/system/anomaly-dashboard.service`**: Hosts the Streamlit UI on port `8501`.

*Note: All three backend services interact with the `machine_telemetry.duckdb` file concurrently. A custom exponential backoff mutex function ensures zero file-lock crashes across processes.*

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now anomaly-upload anomaly-ingest anomaly-dashboard
```

---

## 🧠 Active Learning & Acoustic ML Engine

1. **Acoustic Physics Bootstrap:** On cold-start, the system calculates time-domain metrics (RMS, Peak, Crest Factor, Kurtosis) and spectral metrics (Centroid, Rolloff, Flatness, Sub-band energies) to estimate anomaly types.
2. **On-Device DSP Voice Filter:** Runs real-time harmonic pitch autocorrelation ($80\text{ Hz} - 300\text{ Hz}$) on the Cortex-M33 to suppress human conversation at the edge before recording or alarming.
3. **Continuous Duty & Wear Tracking:** A dynamic Exponential Moving Average (EMA) baseline runs constantly on the MCU. Every 30 seconds, a lightweight JSON heartbeat is sent over MQTT containing the `running_baseline`, `peak_amplitude`, and the machine's ON/OFF state to track acoustic wear trends and calculate duty cycle percentages.
4. **Active Retraining:**
   - Label events in the dashboard (`Mechanical Knock`, `Bearing Wear`, `Cavitation`, `Normal Operation`, etc.).
   - Marking an event as `Normal Operation` instantly calculates and saves a persistent acoustic profile for that specific machine.
   - Click **"🚀 Retrain Model Now"** in the sidebar.
   - A **Scikit-Learn Random Forest Classifier** is trained, and **all unverified historical detections are automatically reclassified** with exact ensemble tree probabilities (`model.predict_proba()`).

---

## 🗺️ Roadmap: Video Capture & Edge Vision Analysis

Because the AMB82-MINI integrates a **Full HD ISP, H.264/H.265 hardware video encoder, and an on-chip NPU (Neural Processing Unit)**, the platform is architected to evolve from acoustic-only detection to **multimodal audio-visual anomaly intelligence**:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ROADMAP MILESTONES                              │
├───────────────────┬────────────────────────────────────────────────────┤
│ 1. Machine ON/OFF │ [x] Continuous baseline acoustic energy profiling    │
│    Duty Tracking  │ [x] Automated operating hour & duty cycle logging    │
│    (COMPLETED)    │ [x] Long-term acoustic volume drift charting         │
├───────────────────┼────────────────────────────────────────────────────┤
│ 2. Synchronized   │ • Enable onboard camera sensor alongside mic       │
│    AV Capture     │ • On acoustic anomaly, record synchronized 5s      │
│                   │   FHD 1080p MP4 video + audio                      │
│                   │ • Stream video directly in dashboard event cards   │
├───────────────────┼────────────────────────────────────────────────────┤
│ 3. Edge NPU       │ • Deploy quantized YOLOv7-Tiny / MobileNet SSD     │
│    Vision AI      │   on the onboard Realtek NPU                       │
│                   │ • Real-time detection of:                          │
│                   │   - Mechanical jams & belt slippage                │
│                   │   - Smoke, fire, or hydraulic fluid leaks          │
│                   │   - Personnel / safety perimeter breaches          │
└───────────────────┴────────────────────────────────────────────────────┘
```

---

## 📦 Enclosures, Cases & Industrial Hardware Protection

### 1. Pre-Built & 3D-Printable Cases
- **3D-Printable Community Enclosures:**
  - Standard STL files for the AMB82-MINI are available on platforms like [Printables](https://www.printables.com) and [Thingiverse](https://www.thingiverse.com) (Search: `AmebaPro2 AMB82-MINI case` or `AMB82-MINI camera mount`).
  - Recommended community designs feature snap-fit lids with dedicated cutouts for the camera sensor, analog microphone port, USB-C power lead, and external IPEX antenna cable routing.
- **Industrial IP65 / IP67 Sealed Polycarbonate Boxes:**
  - For factory environments subject to high particulate, water washdown, or oil mist, standard ABS/Polycarbonate junction boxes (e.g., $100\times68\times50\text{ mm}$ with silicone O-ring gaskets and M12/PG7 cable glands) are ideal.
  - **Acoustic Membrane Protection:** Place an adhesive **hydrophobic/acoustic vent membrane** (such as a GORE acoustic vent sticker) over the microphone aperture hole. This allows acoustic sound waves to pass freely while maintaining a 100% barrier against liquid, grease, and airborne dust.

---

### 2. SoC Resilience: Does the AMB82-MINI Resist Dust & Harsh Environments?

**Yes, significantly better than conventional Single Board Computers (Raspberry Pi, x86 Mini-PCs):**

| Feature | Realtek RTL8735b (AMB82-MINI) | Conventional SBC (Raspberry Pi 4/5) |
|---|---|---|
| **Cooling & Airflow** | **100% Passive (< 1W power draw)** — can be sealed in an airtight, non-ventilated box with zero airflow. | Requires heatsinks, ventilation slots, or active fans that suck airborne industrial dust across components. |
| **Silicon Packaging** | **System-in-Package (SiP)** — CPU, NPU, DSP, Wi-Fi, and RAM are co-packaged in a single sealed BGA chip, eliminating exposed high-speed trace pins. | Discrete high-speed RAM and controller chips with exposed surface solder pins vulnerable to conductive dust bridging. |
| **Storage Medium** | **Solid-State NOR/SPI Flash** — completely resilient to vibration and mechanical wear. | Fragile MicroSD card slots prone to oxidation, dust contamination, or mechanical contact failure. |

#### Recommended Industrial Hardening:
For deployment in environments with conductive metal shavings, high humidity, or oil mist:
1. Mask the microphone pinhole and camera lens with Kapton tape.
2. Apply an **acrylic or silicone conformal coating spray** (e.g., MG Chemicals 419D) over both sides of the PCB.
3. Secure the external IPEX/U.FL antenna connector with a drop of neutral-cure silicone or hot glue to prevent vibration loosening.
