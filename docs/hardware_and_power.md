# 🔬 Hardware Architecture & Low-Power Silicon Alternatives

This document provides a comparative analysis of the **Realtek AmebaPro2 (AMB82-MINI)** against ultra-low-power edge processors, along with electrical power models for battery-powered industrial monitoring.

---

## 1. The Role of the AMB82-MINI in this Project

This project utilizes the **Realtek AmebaPro2 (AMB82-MINI)** as an exceptionally versatile **Proof of Concept (PoC)** and rapid development platform. With dual ARM Cortex-M33 cores, an on-chip NPU (up to 0.4 TOPS), hardware H.264/H.265 video encoding, and integrated Wi-Fi, the AMB82-MINI serves as an all-in-one testbed to validate:
- Edge acoustic signal processing and dynamic noise-floor self-calibration.
- Real-time on-device machine state detection (`IDLE` vs `RUNNING`).
- Synchronized multi-modal capture (audio anomalies paired with future video forensics).

While the AMB82-MINI draws $\approx 150\text{ mA} - 300\text{ mA}$ at 5V ($0.75\text{ W} - 1.5\text{ W}$), making it well-suited for USB or mains-powered industrial equipment, production deployments with strict battery constraints (e.g. multi-year coin cell or solar nodes) can easily slot in lower-power alternatives for acoustic-only monitoring.

---

## 2. Alternative Low-Power Edge Silicon Matrix

| Platform / SoC | Architecture & Key Features | Typical Power Draw | Ideal Use Case |
|---|---|---|---|
| **Realtek AMB82-MINI** *(Current PoC)* | Dual Cortex-M33 + 0.4 TOPS NPU + FHD H.264/H.265 Encoder + Wi-Fi | $\sim 750\text{ mW} - 1.5\text{ W}$ (Active) | Multi-modal Audio + Video Edge AI, mains/USB-powered monitoring |
| **Ambiq Apollo4 Plus / Apollo3** | Sub-Threshold Power Optimization (SPOT®) Cortex-M4/M33 + Low-Power Audio Interface (LPAI) | $< 20\text{ }\mu\text{W}$ (Always-Listening Audio) / $4\text{ }\mu\text{A/MHz}$ | Multi-year CR2032 coin-cell acoustic anomaly sensors |
| **Syntiant NDP120 / NDP101** | Dedicated Ultra-Low-Power Neural Decision Processor + Audio DSP | $< 1\text{ mW}$ (Continuous Neural Inference) | Always-on deep learning acoustic classification coprocessor |
| **Nordic nRF5340 / nRF52840** | Dual Cortex-M33 (nRF5340) with PDM mic interface + BLE 5.4 / Thread / Zigbee | Single-digit $\mu\text{A}$ (Sleep) / $\sim 5\text{ mA}$ (Active Radio) | Battery-operated wireless mesh fleet across warehouse/factory |
| **STM32U5 / STM32H5 Series** | Industrial Cortex-M33 with Autonomous Audio DMA / Multifunction Digital Filter (MDF) | $\sim 110\text{ }\mu\text{A/MHz}$ (Active) / $1.5\text{ }\mu\text{A}$ (Stop mode) | Hardened industrial field sensors (CAN Bus, Modbus RS-485, -40°C to 125°C) |
| **Raspberry Pi RP2350 / RP2040** | Dual Cortex-M33 / Cortex-M0+ with Programmable I/O (PIO) state machines | $\sim 50\text{ mW} - 100\text{ mW}$ | Ultra-low BOM cost ($1 SoC) wired industrial monitoring nodes |

---

## 3. Architectural Portability

Because the signal processing logic developed here relies on general acoustic physics (dual-tier EMA filtering, relative noise-floor hysteresis, spectral flux, and RMS amplitude tracking), the edge algorithms can be compiled down to any C/C++ embedded RTOS environment (FreeRTOS, Zephyr RTOS, or bare-metal ARM CMSIS-DSP) with minimal effort.

---

## 4. Electrical Math: 5-Minute Store-and-Forward Duty-Cycling

To extend battery life on Wi-Fi-enabled hardware without losing telemetry granularity:

### A. Power States (5V USB Input)
* **Wi-Fi Active (TX/RX):** $I_{\text{on}} = 180\text{ mA}$ ($900\text{ mW}$)
* **Wi-Fi Sleeping (Audio Sampling in RAM):** $I_{\text{off}} = 30\text{ mA}$ ($150\text{ mW}$)

### B. Weighted Average Current
In a 5-minute cycle ($300\text{s}$ total):
$$I_{\text{avg}} = \frac{(298\text{ s} \times 30\text{ mA}) + (2\text{ s} \times 180\text{ mA})}{300\text{ s}} = \mathbf{31.0\text{ mA}}$$

This yields an **$83\%$ reduction in average power** ($5.8\times$ battery life extension).

### C. Battery Lifespan Projections (at 80% Usable Capacity)

| Battery Pack | Always-On Wi-Fi ($180\text{ mA}$) | 5-Minute Duty-Cycled ($31\text{ mA}$) |
|---|---|---|
| **5,000 mAh** *(4,000 mAh usable)* | $\approx 22\text{ hours}$ (0.9 days) | **$\approx 129\text{ hours}$ (5.4 days)** |
| **10,000 mAh** *(8,000 mAh usable)* | $\approx 44\text{ hours}$ (1.8 days) | **$\approx 258\text{ hours}$ (10.8 days)** |
| **20,000 mAh** *(16,000 mAh usable)* | $\approx 88\text{ hours}$ (3.7 days) | **$\approx 516\text{ hours}$ (21.5 days)** |
