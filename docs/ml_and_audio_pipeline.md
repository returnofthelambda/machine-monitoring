# 🧠 Active Learning & Acoustic ML Engine

This document details the acoustic physics heuristics, on-device DSP filters, self-calibrating noise engine, and active learning workflow used across the edge nodes and backend server.

---

## 1. Acoustic Physics Bootstrap

On cold start, before supervised models are trained, the system classifies events using physical signal characteristics:

* **Time-Domain Features:**
  * **RMS Amplitude ($A_{\text{rms}}$):** Energy level of the sound clip.
  * **Peak Amplitude ($A_{\text{peak}}$):** Maximum instant value.
  * **Crest Factor ($C = A_{\text{peak}} / A_{\text{rms}}$):** High crest factor indicates sharp impulsive strikes (e.g. `Mechanical Knock`).
  * **Kurtosis:** Statistical fourth moment measuring "tailedness" of the waveform. High kurtosis signifies sudden transients.
* **Frequency-Domain Features:**
  * **Spectral Centroid:** "Center of mass" of the spectrum (measures perceived brightness/shrillness, e.g. `Bearing Wear` screeching).
  * **Spectral Rolloff:** Frequency below which 85% of spectral energy lies.
  * **Spectral Flatness:** Distinguishes noise-like sounds (flat) from tonal sounds (peaked harmonics).
  * **Sub-band Energy Ratios:** Low (0–500 Hz), Mid (500–2500 Hz), High (2500–8000 Hz).

---

## 2. On-Device Voice / Speech Harmonic Filter

To eliminate human conversation from false-triggering industrial anomalies:
* Runs real-time pitch autocorrelation ($80\text{ Hz} - 300\text{ Hz}$) on the Cortex-M33 DSP.
* If periodic human vocal cord pitch is detected, the candidate trigger is suppressed before an anomaly alert or audio capture is generated.

---

## 3. Dual-Tier Self-Calibrating Noise Engine

Rather than requiring manual threshold tuning per room:
* **`ambient_noise_floor`:** A slow minimum-tracking filter continuously identifies the lowest steady background noise floor of the room (e.g., $120$ in a closet, $350$ in a basement near a dehumidifier, or $1200$ on a factory pump floor).
* **`running_baseline`:** A fast Exponential Moving Average (EMA) tracks real-time operating volume.
* **Relative Hysteresis Thresholds:**
  * Startup Threshold: `ambient_noise_floor + max(200, ambient_noise_floor * 0.40)`
  * Shutdown Threshold: `ambient_noise_floor + max(100, ambient_noise_floor * 0.20)`

---

## 4. Active Retraining & Dynamic Reclassification

1. Operators review event cards in the Streamlit UI and label ground truth (`Mechanical Knock`, `Bearing Wear`, `Machine Start / Stop`, `Normal Operation`).
2. Labeling `Normal Operation` computes a persistent ground-truth acoustic baseline vector for that specific machine.
3. Clicking **"🚀 Retrain Model Now"** trains a **Scikit-Learn Random Forest Classifier** on all verified records.
4. The system automatically re-evaluates all unverified historical detections using `model.predict_proba()`, updating predicted labels and confidence scores in DuckDB.

---

## 5. Physical Duration & Running State Health Profiling

* **Transient Duration Envelope ($t_{\text{active}}$):**
  * Rather than relying on static timestamps, the system parses the 25ms sliding RMS energy envelope of the decoded 16kHz audio stream.
  * Calculates dynamic range over the 15th percentile noise floor. Frames exceeding threshold define the precise span and contiguous active duration of the acoustic event.
* **Running State Harmonic Diagnostics:**
  * **Dominant Tone Extraction:** Fast Fourier Transform (FFT) identifies the primary operating harmonic (e.g. 60Hz/120Hz motor poles vs 1.2kHz blade pass frequencies).
  * **3-Band Energy Distribution:** Evaluates low-frequency unbalance (0–500Hz), aerodynamic mid-flow (500–2500Hz), and high-frequency friction/whine (2500–8000Hz).
  * **Stability Metric ($CV$):** Analyzes the coefficient of variation ($\sigma / \mu$) over consecutive running snapshots to distinguish steady constant loads from mechanical surging or hunting.
