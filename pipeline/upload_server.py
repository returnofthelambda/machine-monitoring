import os
import time
import base64
import uuid
import datetime
import duckdb
from flask import Flask, request, jsonify, send_from_directory
from classifier import classify_audio_file

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "audio_samples")
DB_PATH = os.path.join(BASE_DIR, "machine_telemetry.duckdb")

def get_duckdb_connection():
    import time
    for _ in range(20):
        try:
            return duckdb.connect(DB_PATH)
        except duckdb.IOException as e:
            if "lock" in str(e).lower():
                time.sleep(0.2)
            else:
                raise
    return duckdb.connect(DB_PATH)

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Ensure DuckDB table exists with all required columns
con = get_duckdb_connection()
con.execute("""
CREATE TABLE IF NOT EXISTS machine_anomalies (
    event_id VARCHAR PRIMARY KEY,
    machine_id VARCHAR NOT NULL,
    time_detected TIMESTAMP NOT NULL,
    time_stopped TIMESTAMP NOT NULL,
    duration_seconds DOUBLE NOT NULL,
    peak_amplitude DOUBLE,
    audio_path VARCHAR,
    predicted_class VARCHAR,
    confidence DOUBLE,
    verified_label VARCHAR,
    verified_at TIMESTAMP,
    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")
for col, col_type in [
    ("audio_path", "VARCHAR"),
    ("predicted_class", "VARCHAR"),
    ("confidence", "DOUBLE"),
    ("verified_label", "VARCHAR"),
    ("verified_at", "TIMESTAMP")
]:
    try:
        con.execute(f"ALTER TABLE machine_anomalies ADD COLUMN IF NOT EXISTS {col} {col_type};")
    except Exception:
        pass
con.close()

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    """Serve recorded audio files directly over HTTP to any network device."""
    return send_from_directory(UPLOAD_DIR, filename, mimetype="audio/mp4")

@app.route('/upload', methods=['POST'])
def upload_audio():
    """
    Receives Base64 encoded audio from AMB82-MINI, decodes it, saves with unique ID,
    runs ML feature extraction/classification, and updates DuckDB.
    """
    try:
        encoded_data = request.get_data(as_text=True)
        if not encoded_data:
            return jsonify({"error": "No data provided"}), 400

        # Decode the Base64 data
        mp4_data = base64.b64decode(encoded_data)
        if len(mp4_data) == 0:
            return jsonify({"error": "Empty audio payload"}), 400

        # Generate unique timestamped filename
        now = datetime.datetime.now(datetime.timezone.utc)
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        event_uid = str(uuid.uuid4())[:8]
        machine_id = request.headers.get("X-Machine-ID", "MACH_PUMP_01")
        
        filename = f"{machine_id}_{timestamp_str}_{event_uid}.mp4"
        filepath = os.path.join(UPLOAD_DIR, filename)

        with open(filepath, "wb") as f:
            f.write(mp4_data)

        # Also write a symlink/copy for anomaly_audio.mp4 for backwards compatibility
        fixed_path = os.path.join(UPLOAD_DIR, "anomaly_audio.mp4")
        try:
            with open(fixed_path, "wb") as f:
                f.write(mp4_data)
        except Exception:
            pass

        print(f"[UPLOAD] Saved {len(mp4_data)} bytes to {filepath}")

        # 1. Run ML Classification
        predicted_class, confidence = classify_audio_file(filepath)
        print(f"[CLASSIFICATION] {filename} -> {predicted_class} ({confidence*100:.1f}%)")

        # 2. Link with DuckDB Telemetry Event
        # Check if there is an unassociated recent MQTT event for this machine within 60 seconds
        relative_path = os.path.relpath(filepath, os.path.join(BASE_DIR, ".."))
        con = get_duckdb_connection()
        recent_event = con.execute("""
            SELECT event_id FROM machine_anomalies
            WHERE machine_id = ? AND audio_path IS NULL
            ORDER BY logged_at DESC LIMIT 1
        """, (machine_id,)).fetchone()

        if recent_event:
            event_id = recent_event[0]
            con.execute("""
                UPDATE machine_anomalies
                SET audio_path = ?, predicted_class = ?, confidence = ?
                WHERE event_id = ?
            """, (relative_path, predicted_class, confidence, event_id))
            print(f"[DATABASE] Updated event {event_id} with audio and classification.")
        else:
            # Create a new standalone event entry
            event_id = str(uuid.uuid4())
            con.execute("""
                INSERT INTO machine_anomalies (
                    event_id, machine_id, time_detected, time_stopped, duration_seconds, 
                    peak_amplitude, audio_path, predicted_class, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_id,
                machine_id,
                now - datetime.timedelta(seconds=5),
                now,
                5.0,
                0.0,
                relative_path,
                predicted_class,
                confidence
            ))
            print(f"[DATABASE] Created new event {event_id} with audio and classification.")
        con.close()

        return jsonify({
            "status": "success",
            "event_id": event_id,
            "filename": filename,
            "audio_url": f"/audio/{filename}",
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4)
        }), 200

    except Exception as e:
        print(f"[ERROR] Failed to process upload: {e}")
        return jsonify({"error": str(e)}), 500

OTA_DIR = os.path.join(BASE_DIR, "static", "ota")
os.makedirs(OTA_DIR, exist_ok=True)

@app.route('/ota/<path:filename>', methods=['GET'])
@app.route('/api/uploadfile', methods=['GET'])
def serve_ota_file(filename='firmware.bin'):
    target_file = filename if filename else 'firmware.bin'
    if request.path == '/api/uploadfile':
        target_file = 'firmware.bin'
    file_path = os.path.join(OTA_DIR, target_file)
    if not os.path.exists(file_path):
        print(f"[OTA] Requested firmware binary not found: {file_path}")
        return jsonify({"error": f"Firmware binary {target_file} not found"}), 404
    print(f"[OTA] Serving firmware binary {target_file} ({os.path.getsize(file_path)} bytes) to {request.remote_addr}")
    return send_from_directory(OTA_DIR, target_file, as_attachment=True, mimetype='application/octet-stream')

@app.route('/ota/info', methods=['GET'])
def get_ota_info():
    target_file = os.path.join(OTA_DIR, 'firmware.bin')
    if os.path.exists(target_file):
        stat = os.stat(target_file)
        return jsonify({
            "exists": True,
            "filename": "firmware.bin",
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "updated_at": datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        })
    return jsonify({"exists": False, "message": "No firmware.bin staged yet"})

if __name__ == '__main__':
    print(f"Starting upload server on 0.0.0.0:5000, serving audio from {UPLOAD_DIR} and OTA from {OTA_DIR}")
    app.run(host='0.0.0.0', port=5000)
