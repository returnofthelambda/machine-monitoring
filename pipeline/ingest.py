import os
import json
import uuid
import datetime
import duckdb
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "machine_telemetry.duckdb")
MQTT_HOST = os.getenv("MQTT_HOST", "192.168.1.50")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USER = os.getenv("MQTT_USER", "factory_worker")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")

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

CREATE TABLE IF NOT EXISTS machine_telemetry_stream (
    id VARCHAR PRIMARY KEY,
    machine_id VARCHAR NOT NULL,
    machine_state VARCHAR NOT NULL,
    running_baseline DOUBLE,
    peak_amplitude DOUBLE,
    uptime_seconds INTEGER,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

def on_message(client, userdata, msg):
    try:
        topic = msg.topic
        payload = json.loads(msg.payload.decode())
        now = datetime.datetime.now(datetime.timezone.utc)

        conn = get_duckdb_connection()

        if topic.endswith("/telemetry") or "running_baseline" in payload:
            # Periodic Telemetry Heartbeat Stream
            conn.execute("""
                INSERT INTO machine_telemetry_stream (
                    id, machine_id, machine_state, running_baseline, peak_amplitude, uptime_seconds, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                payload.get("machine_id", "UNKNOWN"),
                payload.get("machine_state", "IDLE"),
                float(payload.get("running_baseline", 0.0)),
                float(payload.get("peak_amplitude", 0.0)),
                int(payload.get("uptime_seconds", 0)),
                now
            ))
            print(f"[STREAM] Heartbeat: {payload.get('machine_id')} | State: {payload.get('machine_state')} | Baseline: {payload.get('running_baseline'):.1f}")
        else:
            # Discrete Anomaly Event
            dur = float(payload.get("duration_seconds", 0))
            time_stopped = now
            time_detected = time_stopped - datetime.timedelta(seconds=dur)

            conn.execute("""
                INSERT INTO machine_anomalies (
                    event_id, machine_id, time_detected, time_stopped, duration_seconds, peak_amplitude
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                payload.get("machine_id", "UNKNOWN"),
                time_detected,
                time_stopped,
                dur,
                float(payload.get("peak_amplitude", 0.0))
            ))
            print(f"[ANOMALY] Event recorded for {payload.get('machine_id')} ({dur}s)")

        conn.close()
    except Exception as e:
        print(f"[ERROR] Pipeline parsing failure: {e}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

if MQTT_USER and MQTT_PASSWORD:
    client.username_pw_set(MQTT_USER, MQTT_PASSWORD)

client.on_message = on_message
client.connect(MQTT_HOST, MQTT_PORT, 60)
client.subscribe("factory/+/anomalies")
client.subscribe("factory/+/telemetry")
print(f"Ingestion service connected to {MQTT_HOST}:{MQTT_PORT} (subscribing to anomalies & telemetry streams)")
client.loop_forever()