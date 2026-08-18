# 🏭 Industrial Acoustic Anomaly Detector & Fleet Management

An end-to-end industrial IoT monitoring system powered by the **Realtek AmebaPro2 (AMB82-MINI)** edge AI board, an event-driven **MQTT & HTTP ingestion pipeline**, an **embedded DuckDB analytics store**, and an **interactive Streamlit active-learning dashboard**.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                   AMB82-MINI Edge Sensor Nodes                         │
│                                                                        │
│   • 16kHz Continuous Acoustic Monitoring (ARM Cortex-M33 DSP)          │
│   • Dynamic Self-Calibrating Noise Floor (Dual-Tier EMA Engine)        │
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
                                    │ • Time-Window & Amplitude Filters  │
                                    │ • Human-in-the-Loop Active Retrain │
                                    │ • Scope-Aware Database Wipe Manager│
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

### 3. Initial USB Flash (One-Time Setup)
1. Connect the AMB82-MINI via USB.
2. Enter **Download Mode**: Hold `DOWNLOAD`, press & release `RESET`, then release `DOWNLOAD`.
3. Compile and flash via USB:
   ```bash
   ./build.sh --flash
   ```
4. Press `RESET` once finished. The board is now live on Wi-Fi with OTA enabled.

---

### 4. Over-The-Air (OTA) Wireless Firmware Updates
Once flashed with OTA-enabled firmware, you **never need to plug USB into the board again**.

#### A. Compile & Deploy Firmware to Backend Server
Whenever you make changes to the firmware code:
```bash
./build.sh --deploy-ota
```
This compiles the sketch, generates `build_output/firmware.bin` (~2.7 MB), and uploads it to the `avprox` server at `http://<SERVER_IP>:5000/ota/firmware.bin`.

#### B. Trigger OTA Update on Edge Nodes
You can push the update remotely in two ways:
1. **Via Streamlit Dashboard:** Navigate to the sidebar, open **`📡 Over-The-Air (OTA) Updates`**, and click **`🚀 Push OTA to MACH_PUMP_01`**.
2. **Via MQTT Command Line:**
   ```bash
   mosquitto_pub -h 172.16.40.129 -p 1883 -u factory_worker -P Your_Password \
     -t "factory/MACH_PUMP_01/cmd/ota" \
     -m '{"cmd":"update","resource":"ota/firmware.bin","port":5000}'
   ```
*The board will download the binary over Wi-Fi, flash partition slot B, verify checksums, and reboot into the new firmware in $\approx 8 - 10\text{ seconds}$!*

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
3. **Environment-Agnostic Self-Calibrating Noise Engine:**
   - **`ambient_noise_floor`**: Uses a slow minimum-filter to continuously track the lowest steady background noise floor of the specific room (e.g., $120$ for a quiet closet, $350$ for a basement with a dehumidifier, or $1200$ for a factory pump floor).
   - **`running_baseline`**: Uses a fast Exponential Moving Average (EMA) to track real-time volume.
   - **Relative Thresholding**: `RUNNING` vs `IDLE` state and anomaly triggers automatically calculate relative to *that specific board's local noise floor*, requiring zero manual threshold tuning per device.
4. **Operational State Machine & Spin-Down Capture:**
   - **Hysteresis State Latch:** Prevents rapid re-triggering loops during steady operation. Detects motor startup transients ($IDLE \rightarrow RUNNING$), latches to steady state without false alarms, and detects machine spin-down ($RUNNING \rightarrow IDLE$).
   - **Bi-Directional Audio Capture:** Records 5.0s audio forensics on both machine startup inrush and machine deceleration / spin-down for dynamic mechanical diagnostics.
5. **Ultra-Low-Power Store-and-Forward Wi-Fi Duty-Cycling:**
   - **83% Power Reduction ($5.8\times$ Battery Extension):** Disables the 802.11 Wi-Fi radio for 4m 58s out of every 5 minutes ($I_{\text{avg}} = 31.0\text{ mA}$).
   - **RAM Ring Buffer:** Collects 10 snapshots (1 every 30s) in an internal RAM buffer and flushes them in a single 1.5s MQTT burst to `factory/+/telemetry/batch`.
   - **Instant Anomaly Wakeup:** Machine startup, shutdown, or sudden in-run spikes immediately interrupt sleep, wake Wi-Fi, upload the 5s MP4 audio recording to HTTP `:5000`, publish the anomaly alert, and return to sleep.
6. **Time-Window Filtering & Database Wipe Manager:**
   - Filter telemetry trends and events by time range (`Last 1 Hour` to `Last 7 Days`) and minimum amplitude threshold.
   - Scope-aware database manager to wipe continuous stream logs, delete `.mp4` audio archives, or execute a full factory database purge (`DELETE FROM machine_telemetry_stream, machine_anomalies, machine_baselines`).
7. **High-Performance Paginated Dashboard UI:**
   - **Event Card Pagination:** Slices historical event cards (15 per page) to ensure sub-50ms render times even with thousands of records in DuckDB.
   - **On-Demand Manual Sync & `@st.fragment` Auto-Sync:** Instant `⚡ Sync` button plus optional non-blocking background chart updates without browser freezing or interrupting audio playback.
8. **Active Learning & Automated Model Retraining:**
   - Label events in the dashboard (`Mechanical Knock`, `Bearing Wear`, `Machine Start / Stop`, `Normal Operation`, etc.).
   - Marking an event as `Normal Operation` instantly calculates and saves a persistent ground-truth acoustic baseline vector for that specific machine.
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

## 🔬 Proof of Concept & Low-Power Hardware Alternatives

### The Role of the AMB82-MINI in this Project
This project uses the **Realtek AmebaPro2 (AMB82-MINI)** as an exceptionally versatile **Proof of Concept (PoC)** and prototyping platform. With dual ARM Cortex-M33 cores, an on-chip NPU (up to 0.4 TOPS), hardware H.264/H.265 video encoding, and integrated Wi-Fi, the AMB82-MINI serves as an all-in-one testbed to validate:
- Edge acoustic signal processing and dynamic noise-floor self-calibration.
- Real-time on-device machine state detection (`IDLE` vs `RUNNING`).
- Synchronized multi-modal capture (audio anomalies paired with future video forensics).

While the AMB82-MINI draws $\approx 150\text{ mA} - 300\text{ mA}$ at 5V ($0.75\text{ W} - 1.5\text{ W}$), making it well-suited for USB or mains-powered industrial equipment, production deployments with strict battery constraints (e.g. multi-year coin cell or solar nodes) can easily slot in lower-power alternatives for acoustic-only monitoring.

---

### Alternative Low-Power Edge Silicon Options

| Platform / SoC | Architecture & Key Features | Typical Power Draw | Ideal Use Case |
|---|---|---|---|
| **Realtek AMB82-MINI** *(Current PoC)* | Dual Cortex-M33 + 0.4 TOPS NPU + FHD H.264/H.265 Encoder + Wi-Fi | $\sim 750\text{ mW} - 1.5\text{ W}$ (Active) | Multi-modal Audio + Video Edge AI, mains/USB-powered monitoring |
| **Ambiq Apollo4 Plus / Apollo3** | Sub-Threshold Power Optimization (SPOT®) Cortex-M4/M33 + Low-Power Audio Interface (LPAI) | $< 20\text{ }\mu\text{W}$ (Always-Listening Audio) / $4\text{ }\mu\text{A/MHz}$ | Multi-year CR2032 coin-cell acoustic anomaly sensors |
| **Syntiant NDP120 / NDP101** | Dedicated Ultra-Low-Power Neural Decision Processor + Audio DSP | $< 1\text{ mW}$ (Continuous Neural Inference) | Always-on deep learning acoustic classification coprocessor |
| **Nordic nRF5340 / nRF52840** | Dual Cortex-M33 (nRF5340) with PDM mic interface + BLE 5.4 / Thread / Zigbee | Single-digit $\mu\text{A}$ (Sleep) / $\sim 5\text{ mA}$ (Active Radio) | Battery-operated wireless mesh fleet across warehouse/factory |
| **STM32U5 / STM32H5 Series** | Industrial Cortex-M33 with Autonomous Audio DMA / Multifunction Digital Filter (MDF) | $\sim 110\text{ }\mu\text{A/MHz}$ (Active) / $1.5\text{ }\mu\text{A}$ (Stop mode) | Hardened industrial field sensors (CAN Bus, Modbus RS-485, -40°C to 125°C) |
| **Raspberry Pi RP2350 / RP2040** | Dual Cortex-M33 / Cortex-M0+ with Programmable I/O (PIO) state machines | $\sim 50\text{ mW} - 100\text{ mW}$ | Ultra-low BOM cost ($1 SoC) wired industrial monitoring nodes |

#### Architectural Portability
Because the signal processing logic developed here relies on general acoustic physics (dual-tier EMA filtering, relative noise-floor hysteresis, spectral flux, and RMS amplitude tracking), the edge algorithms can be compiled down to any C/C++ embedded RTOS environment (FreeRTOS, Zephyr RTOS, or bare-metal ARM CMSIS-DSP) with minimal effort.

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
