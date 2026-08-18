# 🗺️ Roadmap: Multi-Modal Audio-Visual & Edge Vision Analysis

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

## Technical Integration Details

1. **Hardware Stream Linking:**
   * Audio: `Audio -> AAC Encoder -> MP4 Muxer`
   * Video: `Camera Sensor -> ISP -> H.264 Encoder -> MP4 Muxer`
   * AmebaPro2 StreamIO subsystem automatically synchronizes audio/video timestamps in the hardware container.

2. **Edge NPU Inference Pipeline:**
   * Quantized NN models loaded directly from SPI Flash / SD Card into NPU RAM.
   * Runs frame-by-frame inference concurrently with acoustic DSP on the Cortex-M33 cores.
