import os
import av
import numpy as np
import scipy.signal as signal
from scipy.fft import rfftfreq
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import duckdb

def get_duckdb_connection(db_path, read_only=False):
    import time
    for _ in range(20):
        try:
            return duckdb.connect(db_path, read_only=read_only)
        except duckdb.IOException as e:
            if "lock" in str(e).lower():
                time.sleep(0.2)
            else:
                raise
    return duckdb.connect(db_path, read_only=read_only)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "acoustic_classifier.joblib")
os.makedirs(MODEL_DIR, exist_ok=True)

DEFAULT_CLASSES = [
    "Normal Operation",
    "Bearing Wear / Friction",
    "Cavitation / Flow Turbulence",
    "Mechanical Knock / Impact",
    "Human Speech / Conversation",
    "Ambient / Background Noise"
]

def decode_audio_from_mp4(file_path):
    """Decodes audio from an MP4/AAC file into a float32 numpy array and sample rate."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    container = av.open(file_path)
    audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
    if not audio_stream:
        raise ValueError(f"No audio stream found in {file_path}")

    samples = []
    sample_rate = audio_stream.codec_context.sample_rate or 16000
    for frame in container.decode(audio_stream):
        arr = frame.to_ndarray()
        if arr.ndim > 1:
            arr = arr.mean(axis=0)  # convert stereo to mono
        samples.append(arr)

    if not samples:
        return np.zeros(16000, dtype=np.float32), sample_rate

    audio_data = np.concatenate(samples).astype(np.float32)
    if np.issubdtype(samples[0].dtype, np.integer):
        audio_data = audio_data / 32768.0

    return audio_data, sample_rate

def extract_features(audio_data, sample_rate=16000):
    """
    Extracts comprehensive acoustic features from audio waveform:
    - Time-domain: RMS, Peak, Crest Factor, Zero-crossing rate, Kurtosis
    - Spectral-domain: Spectral Centroid, Rolloff, Spread, Flatness, Sub-band energies
    - Voice/Speech Discrimination: Pitch periodicity (80-300Hz), Voice band ratio (300-3400Hz), Spectral Flux
    """
    if len(audio_data) == 0:
        return np.zeros(23, dtype=np.float32)

    # Clean signal
    y = np.nan_to_num(audio_data)
    if np.max(np.abs(y)) > 0:
        y = y / (np.max(np.abs(y)) + 1e-8)  # normalize

    # 1. Time-Domain Metrics
    rms = np.sqrt(np.mean(y**2) + 1e-12)
    peak = np.max(np.abs(y))
    crest_factor = peak / (rms + 1e-8)
    zcr = np.mean(np.abs(np.diff(np.signbit(y))))
    kurtosis = np.mean((y - np.mean(y))**4) / (np.var(y)**2 + 1e-8)

    # 2. Spectral Analysis (STFT)
    n_fft = 2048
    hop_length = 512
    freqs = rfftfreq(n_fft, d=1.0/sample_rate)
    
    f, t, Zxx = signal.stft(y, fs=sample_rate, nperseg=n_fft, noverlap=n_fft-hop_length)
    mag_spec = np.abs(Zxx)
    mean_spec = np.mean(mag_spec, axis=1) + 1e-12
    total_energy = np.sum(mean_spec)

    # Spectral Centroid & Spread
    centroid = np.sum(freqs * mean_spec) / total_energy
    spread = np.sqrt(np.sum(((freqs - centroid)**2) * mean_spec) / total_energy)
    
    # Spectral Rolloff (85% energy point)
    cum_energy = np.cumsum(mean_spec)
    rolloff_idx = np.searchsorted(cum_energy, 0.85 * total_energy)
    rolloff = freqs[min(rolloff_idx, len(freqs)-1)]

    # Spectral Flatness (Wiener entropy)
    geom_mean = np.exp(np.mean(np.log(mean_spec)))
    arith_mean = np.mean(mean_spec)
    flatness = geom_mean / (arith_mean + 1e-8)

    # Sub-band Energies (8 frequency bands up to Nyquist)
    band_edges = [0, 250, 500, 1000, 2000, 3000, 4000, 6000, sample_rate // 2]
    band_energies = []
    for i in range(len(band_edges) - 1):
        idx_low = np.searchsorted(freqs, band_edges[i])
        idx_high = np.searchsorted(freqs, band_edges[i+1])
        band_energy = np.sum(mean_spec[idx_low:idx_high]) / total_energy
        band_energies.append(band_energy)

    # 3. Human Speech & Conversation Discrimination Features
    # Pitch Periodicity (Autocorrelation in 80Hz-300Hz range)
    n_corr = min(len(y), sample_rate * 2)
    sig_center = y[:n_corr] - np.mean(y[:n_corr])
    corr = np.correlate(sig_center, sig_center, mode='full')
    corr = corr[len(corr)//2:]
    
    lag_min = int(sample_rate / 300) # ~53 samples (300Hz)
    lag_max = int(sample_rate / 80)  # ~200 samples (80Hz)
    if lag_max < len(corr) and corr[0] > 0:
        pitch_peak = float(np.max(corr[lag_min:lag_max]) / corr[0])
    else:
        pitch_peak = 0.0

    # Voice Formant Band Ratio (300Hz - 3400Hz)
    idx_v_low = np.searchsorted(freqs, 300)
    idx_v_high = np.searchsorted(freqs, 3400)
    voice_band_ratio = float(np.sum(mean_spec[idx_v_low:idx_v_high]) / total_energy)

    # Spectral Flux (Dynamic phoneme transitions)
    norm_mag = mag_spec / (np.sum(mag_spec, axis=0, keepdims=True) + 1e-8)
    spectral_flux = float(np.mean(np.sqrt(np.sum(np.diff(norm_mag, axis=1)**2, axis=0))))

    features = [
        rms,
        peak,
        crest_factor,
        zcr,
        kurtosis,
        centroid / (sample_rate / 2.0),
        spread / (sample_rate / 2.0),
        rolloff / (sample_rate / 2.0),
        flatness,
        pitch_peak,
        voice_band_ratio,
        spectral_flux,
        *band_energies
    ]
    return np.array(features, dtype=np.float32)

def heuristic_predict(features):
    """
    Calibrated heuristic prediction when insufficient human labels exist.
    """
    rms, peak, crest_factor, zcr, kurtosis, norm_centroid, norm_spread, norm_rolloff, flatness = features[:9]
    pitch_peak = features[9]
    voice_band_ratio = features[10]
    spectral_flux = features[11]
    
    high_freq_energy = np.sum(features[16:])  # > 2kHz
    low_freq_energy = np.sum(features[12:15])   # < 1kHz

    # 1. Human Speech / Conversation Detection (Harmonic pitch + 300-3400Hz formant concentration)
    if pitch_peak > 0.35 and voice_band_ratio > 0.50 and crest_factor < 6.0:
        return "Human Speech / Conversation", float(min(0.95, 0.65 + pitch_peak * 0.35))

    # 2. High crest factor / kurtosis -> Mechanical Knock / Impact
    if crest_factor > 4.5 or kurtosis > 8.0:
        return "Mechanical Knock / Impact", float(min(0.92, 0.55 + 0.05 * crest_factor))
    
    # 3. High frequency noise + high zero crossings -> Bearing Wear / Friction
    if norm_centroid > 0.45 and high_freq_energy > 0.40 and zcr > 0.15:
        return "Bearing Wear / Friction", float(min(0.88, 0.50 + high_freq_energy * 0.4))

    # 4. Broad spectrum + high spectral flatness -> Cavitation / Flow Turbulence
    if flatness > 0.25 and norm_spread > 0.30:
        return "Cavitation / Flow Turbulence", float(min(0.85, 0.50 + flatness))

    # 5. Mostly low frequency / gentle continuous rumble -> Normal Operation
    if low_freq_energy > 0.65:
        return "Normal Operation", float(min(0.89, 0.55 + low_freq_energy * 0.35))

    return "Ambient / Background Noise", 0.60

def classify_audio_file(audio_path):
    """
    Classifies an audio file using either the trained ML model (if available)
    or the acoustic heuristic engine.
    """
    try:
        audio_data, sr = decode_audio_from_mp4(audio_path)
        feat = extract_features(audio_data, sr)

        if os.path.exists(MODEL_PATH):
            try:
                pipeline = joblib.load(MODEL_PATH)
                proba = pipeline.predict_proba([feat])[0]
                classes = pipeline.classes_
                best_idx = np.argmax(proba)
                return str(classes[best_idx]), float(proba[best_idx])
            except Exception as e:
                print(f"[CLASSIFIER] Model inference failed, using fallback: {e}")

        # Fallback to acoustic heuristics
        predicted_class, confidence = heuristic_predict(feat)
        return str(predicted_class), float(confidence)
    except Exception as e:
        print(f"[CLASSIFIER] Feature extraction failure on {audio_path}: {e}")
        return "Unknown", 0.0

def retrain_model_from_database(db_path="machine_telemetry.duckdb"):
    """
    Queries verified labels from DuckDB, extracts features from associated audio files,
    trains a RandomForest pipeline, and saves the serialized model.
    """
    if not os.path.exists(db_path):
        return {"status": "error", "message": "Database not found."}

    con = get_duckdb_connection(db_path, read_only=True)
    rows = con.execute("""
        SELECT audio_path, verified_label
        FROM machine_anomalies
        WHERE verified_label IS NOT NULL 
          AND verified_label NOT IN ('', 'Ignored / False Positive', 'Non-Classified', 'Unknown')
          AND audio_path IS NOT NULL
    """).fetchall()
    con.close()

    if len(rows) < 2:
        return {
            "status": "warning",
            "message": f"Need at least 2 verified labels to train. Current count: {len(rows)}"
        }

    X = []
    y = []
    for audio_path, label in rows:
        resolved_path = audio_path
        if not os.path.exists(resolved_path):
            alt_path = os.path.join(os.path.dirname(__file__), "..", audio_path)
            if os.path.exists(alt_path):
                resolved_path = alt_path
            else:
                continue

        try:
            audio_data, sr = decode_audio_from_mp4(resolved_path)
            feat = extract_features(audio_data, sr)
            X.append(feat)
            y.append(label)
        except Exception as e:
            print(f"[RETRAIN] Skipping invalid audio file {resolved_path}: {e}")

    unique_classes = set(y)
    if len(unique_classes) < 2:
        return {
            "status": "warning",
            "message": f"Labels must contain at least 2 distinct classes. Found: {list(unique_classes)}"
        }

    X = np.array(X)
    y = np.array(y)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, min_samples_leaf=1))
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)

    # Re-infer and update predictions on all unverified items with the newly trained model
    con = get_duckdb_connection(db_path)
    unverified_rows = con.execute("""
        SELECT event_id, audio_path
        FROM machine_anomalies
        WHERE (verified_label IS NULL OR verified_label IN ('', 'Non-Classified'))
          AND audio_path IS NOT NULL
    """).fetchall()

    reclassified_count = 0
    for event_id, audio_path in unverified_rows:
        resolved_path = audio_path
        if not os.path.exists(resolved_path):
            alt_path = os.path.join(os.path.dirname(__file__), "..", audio_path)
            if os.path.exists(alt_path):
                resolved_path = alt_path
            else:
                alt2 = os.path.join(os.path.dirname(__file__), "audio_samples", os.path.basename(audio_path))
                if os.path.exists(alt2):
                    resolved_path = alt2
                else:
                    continue

        try:
            audio_data, sr = decode_audio_from_mp4(resolved_path)
            feat = extract_features(audio_data, sr)
            proba = pipeline.predict_proba([feat])[0]
            best_idx = np.argmax(proba)
            pred_class = str(pipeline.classes_[best_idx])
            conf = float(proba[best_idx])

            con.execute("""
                UPDATE machine_anomalies
                SET predicted_class = ?, confidence = ?
                WHERE event_id = ?
            """, (pred_class, conf, event_id))
            reclassified_count += 1
        except Exception as e:
            print(f"[RECLASSIFY] Error reclassifying event {event_id}: {e}")

    con.close()

    return {
        "status": "success",
        "samples": len(X),
        "classes": list(unique_classes),
        "reclassified": reclassified_count,
        "message": f"Successfully retrained model on {len(X)} verified samples and updated predictions on {reclassified_count} unverified items across {len(unique_classes)} classes."
    }

def update_machine_baseline(machine_id, db_path="machine_telemetry.duckdb"):
    """
    Computes and saves an acoustic baseline profile for a machine based on
    all audio samples verified as 'Normal Operation'.
    """
    if not os.path.exists(db_path):
        return None

    con = get_duckdb_connection(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS machine_baselines (
            machine_id VARCHAR PRIMARY KEY,
            baseline_rms DOUBLE,
            baseline_peak DOUBLE,
            baseline_centroid DOUBLE,
            sample_count INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    rows = con.execute("""
        SELECT audio_path FROM machine_anomalies
        WHERE machine_id = ? AND verified_label = 'Normal Operation' AND audio_path IS NOT NULL
    """, (machine_id,)).fetchall()

    if not rows:
        con.close()
        return None

    rms_list = []
    peak_list = []
    centroid_list = []

    for (audio_path,) in rows:
        resolved_path = audio_path
        if not os.path.exists(resolved_path):
            alt = os.path.join(os.path.dirname(__file__), "..", audio_path)
            alt2 = os.path.join(os.path.dirname(__file__), "audio_samples", os.path.basename(audio_path))
            if os.path.exists(alt): resolved_path = alt
            elif os.path.exists(alt2): resolved_path = alt2
            else: continue

        try:
            audio_data, sr = decode_audio_from_mp4(resolved_path)
            feat = extract_features(audio_data, sr)
            rms_list.append(float(feat[0]))
            peak_list.append(float(feat[1]))
            centroid_list.append(float(feat[5]))
        except Exception:
            pass

    if rms_list:
        mean_rms = float(np.mean(rms_list))
        mean_peak = float(np.mean(peak_list))
        mean_cent = float(np.mean(centroid_list))
        count = len(rms_list)

        con.execute("""
            INSERT INTO machine_baselines (machine_id, baseline_rms, baseline_peak, baseline_centroid, sample_count, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (machine_id) DO UPDATE SET
                baseline_rms = excluded.baseline_rms,
                baseline_peak = excluded.baseline_peak,
                baseline_centroid = excluded.baseline_centroid,
                sample_count = excluded.sample_count,
                updated_at = CURRENT_TIMESTAMP;
        """, (machine_id, mean_rms, mean_peak, mean_cent, count))

    con.close()
    return {"machine_id": machine_id, "sample_count": len(rms_list)}

def get_machine_baseline(machine_id, db_path="machine_telemetry.duckdb"):
    """Fetches the saved acoustic baseline for a given machine."""
    if not os.path.exists(db_path):
        return None
    con = get_duckdb_connection(db_path, read_only=True)
    try:
        row = con.execute("""
            SELECT baseline_rms, baseline_peak, baseline_centroid, sample_count, updated_at
            FROM machine_baselines
            WHERE machine_id = ?
        """, (machine_id,)).fetchone()
        con.close()
        if row:
            return {
                "baseline_rms": row[0],
                "baseline_peak": row[1],
                "baseline_centroid": row[2],
                "sample_count": row[3],
                "updated_at": row[4]
            }
    except Exception:
        con.close()
    return None

