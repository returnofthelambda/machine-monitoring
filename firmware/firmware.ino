#include <ArduinoJson.h>
#include <PubSubClient.h>
#include <WiFi.h>
#include "AudioStream.h"
#include "StreamIO.h"
#include "AudioEncoder.h"
#include "MP4Recording.h"
#include "AmebaFatFS.h"
#include "Base64.h"

// Realtek SDK C API to disable the built-in auto-reconnect thread and handle OTA
extern "C" {
  int wifi_config_autoreconnect(unsigned char, unsigned char, unsigned short);
  int http_update_ota(char *host, int port, char *resource);
  void ota_platform_reset(void);
}

#ifndef WIFI_SSID
#define WIFI_SSID "FALLBACK_SSID"
#endif
#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD "FALLBACK_PASSWORD"
#endif
#ifndef MQTT_HOST
#define MQTT_HOST "192.168.1.50"
#endif
#ifndef MQTT_PORT
#define MQTT_PORT 1883
#endif
#ifndef UPLOAD_HOST
#define UPLOAD_HOST "192.168.1.100"
#endif
#ifndef UPLOAD_PORT
#define UPLOAD_PORT 5000
#endif
#ifndef MQTT_USER
#define MQTT_USER "factory_worker"
#endif
#ifndef MQTT_PASSWORD
#define MQTT_PASSWORD "FALLBACK_PASS"
#endif
#ifndef MQTT_TOPIC
#define MQTT_TOPIC "factory/machine_01/anomalies"
#endif
#ifndef MACHINE_ID
#define MACHINE_ID "MACH_PUMP_01"
#endif

const int RMS_THRESHOLD = 550;
const unsigned long COOLDOWN_MS = 1500;

class CustomAudio : public Audio {
public:
    void setLeftMicCallback(void (*cb)(const uint8_t *, int, uint8_t, uint32_t, audio_params_t)) {
        if (_p_mmf_context) {
            mm_module_ctrl(_p_mmf_context, CMD_AUDIO_SET_LEFTMIC_CB, (int)cb);
        }
    }
};

AudioSetting configA(1); // 16kHz Mono Analog Mic
CustomAudio audio;

static volatile int latest_audio_val = 0;
static volatile bool latest_is_voice = false;

static void audio_rx_callback(const uint8_t *pbuf, int size, uint8_t ch, uint32_t ts, audio_params_t params) {
  int16_t *samples = (int16_t *)pbuf;
  int num_samples = size / sizeof(int16_t);
  int peak = 0;
  long long sum_sq = 0;

  for (int i = 0; i < num_samples; i++) {
    int val = abs((int)samples[i]);
    if (val > peak) {
      peak = val;
    }
    sum_sq += (long long)samples[i] * samples[i];
  }
  latest_audio_val = peak;

  // On-device Voice Discrimination (Harmonic Pitch Autocorrelation)
  if (peak > RMS_THRESHOLD && num_samples >= 256) {
    int n_eval = num_samples - 200;
    long long energy0 = 0;
    for (int i = 0; i < n_eval; i += 2) {
      energy0 += (long long)samples[i] * samples[i];
    }

    long long max_pitch_corr = 0;
    if (energy0 > 0) {
      // Vocal pitch frequency search 80Hz - 300Hz (lag 54 to 198 at 16kHz)
      for (int lag = 54; lag <= 198; lag += 4) {
        long long corr = 0;
        for (int i = 0; i < n_eval; i += 4) {
          corr += (long long)samples[i] * samples[i + lag];
        }
        if (corr > max_pitch_corr) {
          max_pitch_corr = corr;
        }
      }
      float pitch_ratio = (float)max_pitch_corr / (float)(energy0 / 2);
      float rms = sqrt((double)sum_sq / num_samples);
      float crest = (float)peak / (rms + 1e-5f);

      // Voice profile: periodic pitch harmonics (> 0.40) + non-impulsive crest (< 5.0)
      if (pitch_ratio > 0.40f && crest < 5.0f) {
        latest_is_voice = true;
      } else {
        latest_is_voice = false;
      }
    } else {
      latest_is_voice = false;
    }
  } else {
    latest_is_voice = false;
  }
}

void init_onboard_mic() {
  audio.configAudio(configA);
  audio.begin();
  audio.setAMicBoost(MIC_20DB);
  audio.setMicGain(50);
  audio.setLeftMicCallback(audio_rx_callback);
}

AmebaFatFS fs;
AAC aac;
MP4Recording mp4;
StreamIO audioStreamer1(1, 1);
StreamIO audioStreamer2(1, 1);

#define FILENAME "anomaly_audio.mp4"
int recordingstate = -1;
int previousRecordingState = -1;

WiFiClient httpClient;

void encodeMP4andsendHttpPostRequest() {
    String filepath = String(fs.getRootPath()) + FILENAME;
    File file = fs.open(filepath);
    if (!file) {
       Serial.println("Failed to open mp4 for reading");
       return;
    }
    unsigned int fileSize = file.size();
    if (fileSize == 0) {
       Serial.println("MP4 file is empty (0 bytes), skipping upload.");
       file.close();
       return;
    }
    Serial.print("Reading MP4 file: ");
    Serial.print(fileSize);
    Serial.println(" bytes");

    uint8_t *fileinput = (uint8_t *)malloc(fileSize + 1);
    if (!fileinput) {
       Serial.println("Failed to allocate memory for MP4 file");
       file.close();
       return;
    }
    file.read(fileinput, fileSize);
    fileinput[fileSize] = '\0';
    file.close();

    int encodedLen = base64_enc_len(fileSize);
    char *encodedData = (char *)malloc(encodedLen + 1);
    if (!encodedData) {
       Serial.println("Failed to allocate memory for Base64 buffer");
       free(fileinput);
       return;
    }
    int actualEncLen = base64_encode(encodedData, (char *)fileinput, fileSize);
    free(fileinput);

    Serial.print("Connecting to upload server: ");
    Serial.print(UPLOAD_HOST);
    Serial.print(":");
    Serial.println(UPLOAD_PORT);

    if (httpClient.connect(UPLOAD_HOST, UPLOAD_PORT)) {
        Serial.println("Connected to upload server! Sending HTTP POST...");
        httpClient.println("POST /upload HTTP/1.1");
        httpClient.print("Host: ");
        httpClient.println(UPLOAD_HOST);
        httpClient.println("Content-Type: application/x-www-form-urlencoded");
        httpClient.print("Content-Length: ");
        httpClient.println(actualEncLen);
        httpClient.print("X-Machine-ID: ");
        httpClient.println(MACHINE_ID);
        httpClient.println("Connection: close");
        httpClient.println(); 
        
        // Write encoded data
        httpClient.print(encodedData);
        Serial.println("Data sent, waiting for server response...");

        // Wait for server response
        unsigned long timeout = millis();
        while (httpClient.connected() && millis() - timeout < 5000) {
            while (httpClient.available()) {
                char c = httpClient.read();
                Serial.write(c);
                timeout = millis();
            }
        }
        Serial.println("\nHTTP upload transaction complete.");
        httpClient.stop();
    } else {
        Serial.print("Failed to connect to upload server at ");
        Serial.print(UPLOAD_HOST);
        Serial.print(":");
        Serial.println(UPLOAD_PORT);
    }
    free(encodedData);
}



WiFiClient netClient;
PubSubClient client(netClient);

bool is_anomalous = false;
unsigned long anomaly_start_ms = 0;
unsigned long last_loud_ms = 0;
int peak_val = 0;

void setup_wifi() {
  Serial.println("\n--- Initializing WiFi ---");

  int status = WL_IDLE_STATUS;
  int attempt = 0;
  while (status != WL_CONNECTED) {
    attempt++;
    Serial.print("Connecting to WiFi (attempt ");
    Serial.print(attempt);
    Serial.print("): SSID=[");
    Serial.print(WIFI_SSID);
    Serial.println("]");
    status = WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    if (status != WL_CONNECTED) {
      Serial.println("Connection failed. Retrying in 10s...");
      delay(10000);
    }
  }
  Serial.print("WiFi connected! IP: ");
  Serial.println(WiFi.localIP());
}

void trigger_ota_update(const char* host, int port, const char* resource_path) {
  Serial.print("\n[OTA] ========================================\n");
  Serial.print("[OTA] Starting OTA Firmware Download from: http://");
  Serial.print(host);
  Serial.print(":");
  Serial.print(port);
  Serial.print("/");
  Serial.println(resource_path);
  Serial.println("[OTA] Target flash: Partition Slot B");

  // Unlink MP4 recording to free resources during firmware flash
  if (mp4.getRecordingState() == 1) {
    mp4.begin();
  }

  int ret = http_update_ota((char*)host, port, (char*)resource_path);
  if (ret == 0) {
    Serial.println("[OTA] Firmware successfully downloaded and verified!");
    Serial.println("[OTA] Rebooting into updated firmware slot...");
    delay(1000);
    ota_platform_reset();
  } else {
    Serial.print("[OTA] ERROR: OTA Update failed with return code: ");
    Serial.println(ret);
  }
}

void mqtt_callback(char* topic, byte* payload, unsigned int length) {
  char msg[256];
  if (length >= sizeof(msg)) length = sizeof(msg) - 1;
  memcpy(msg, payload, length);
  msg[length] = '\0';

  Serial.print("[MQTT] Received command on topic [");
  Serial.print(topic);
  Serial.print("]: ");
  Serial.println(msg);

  if (String(topic).endsWith("/cmd/ota")) {
    JsonDocument doc;
    DeserializationError err = deserializeJson(doc, msg);
    const char* resource = "ota/firmware.bin";
    const char* host = UPLOAD_HOST;
    int port = UPLOAD_PORT;

    if (!err) {
      if (doc.containsKey("resource")) resource = doc["resource"];
      if (doc.containsKey("host")) host = doc["host"];
      if (doc.containsKey("port")) port = doc["port"];
    }

    trigger_ota_update(host, port, resource);
  }
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection as ");
    Serial.print(MACHINE_ID);
    Serial.print(" (user: ");
    Serial.print(MQTT_USER);
    Serial.print(")... ");
    if (client.connect(MACHINE_ID, MQTT_USER, MQTT_PASSWORD)) {
      Serial.println("connected!");
      String otaTopic = "factory/" + String(MACHINE_ID) + "/cmd/ota";
      client.subscribe(otaTopic.c_str());
      client.subscribe("factory/all/cmd/ota");
      Serial.print("[MQTT] Subscribed to OTA command topics: ");
      Serial.print(otaTopic);
      Serial.println(" and factory/all/cmd/ota");
      break;
    } else {
      Serial.print("failed, rc=");
      Serial.println(client.state());
      delay(2000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n--- BOOTING ---");
  
  setup_wifi();
  Serial.println("WiFi setup complete.");
  
  Serial.println("Initializing FS...");
  fs.begin();
  Serial.println("FS initialized.");

  client.setServer(MQTT_HOST, MQTT_PORT);
  client.setCallback(mqtt_callback);
  
  Serial.println("Initializing Mic...");
  init_onboard_mic();
  Serial.println("Mic initialized.");

  Serial.println("Initializing AAC...");
  aac.configAudio(configA);
  aac.begin();
  Serial.println("AAC initialized.");

  Serial.println("Initializing MP4...");
  mp4.configAudio(configA, CODEC_AAC);
  mp4.setRecordingDuration(5);
  mp4.setRecordingFileCount(1);
  mp4.setRecordingFileName("anomaly_audio");
  mp4.setRecordingDataType(STORAGE_AUDIO);
  Serial.println("MP4 initialized.");

  Serial.println("Linking StreamIO 1...");
  audioStreamer1.registerInput(audio);
  audioStreamer1.registerOutput(aac);
  if (audioStreamer1.begin() != 0) {
      Serial.println("StreamIO link 1 start failed");
  }

  Serial.println("Linking StreamIO 2...");
  audioStreamer2.registerInput(aac);
  audioStreamer2.registerOutput(mp4);
  if (audioStreamer2.begin() != 0) {
      Serial.println("StreamIO link 2 start failed");
  }

  Serial.println("Audio detector initialized.");

  // Send initial boot beacon over Wi-Fi, then sleep
  if (ensure_network_connected()) {
    JsonDocument tdoc;
    tdoc["machine_id"] = MACHINE_ID;
    tdoc["machine_state"] = "IDLE";
    tdoc["running_baseline"] = 0.0f;
    tdoc["peak_amplitude"] = 0;
    tdoc["uptime_seconds"] = 0;
    char tbuf[256];
    serializeJson(tdoc, tbuf);
    String telemetryTopic = "factory/" + String(MACHINE_ID) + "/telemetry";
    client.publish(telemetryTopic.c_str(), tbuf);
  }
  disconnect_wifi_and_sleep();
}

// --- 5-MINUTE STORE-AND-FORWARD TELEMETRY BUFFER ---
struct TelemetrySnapshot {
  char state[8];
  float baseline;
  int peak;
  int uptime;
};

#define BATCH_BUFFER_MAX 10
TelemetrySnapshot telemetry_batch[BATCH_BUFFER_MAX];
int telemetry_batch_count = 0;
unsigned long last_batch_flush_ms = 0;
const unsigned long BATCH_FLUSH_INTERVAL_MS = 300000; // 5 minutes (300s)

bool ensure_network_connected() {
  if (WiFi.status() == WL_CONNECTED && client.connected()) {
    return true;
  }

  Serial.println("[POWER] Waking Wi-Fi radio & connecting to network...");
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 5) {
    attempts++;
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    unsigned long start = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - start < 4000) {
      delay(100);
    }
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[POWER] Wi-Fi connection FAILED");
    return false;
  }

  Serial.print("[POWER] Wi-Fi connected! IP: ");
  Serial.println(WiFi.localIP());

  // Connect to MQTT Broker
  attempts = 0;
  while (!client.connected() && attempts < 3) {
    attempts++;
    if (client.connect(MACHINE_ID, MQTT_USER, MQTT_PASSWORD)) {
      Serial.println("[POWER] MQTT Connected!");
      String otaTopic = "factory/" + String(MACHINE_ID) + "/cmd/ota";
      client.subscribe(otaTopic.c_str());
      client.subscribe("factory/all/cmd/ota");
      break;
    }
    delay(500);
  }

  return client.connected();
}

void disconnect_wifi_and_sleep() {
  Serial.println("[POWER] Disconnecting and powering down Wi-Fi radio (30-min sleep)...");
  if (client.connected()) {
    client.disconnect();
  }
  WiFi.disconnect();
}

void flush_telemetry_batch_and_sleep() {
  if (telemetry_batch_count == 0) return;

  if (ensure_network_connected()) {
    Serial.print("[POWER] Flushing ");
    Serial.print(telemetry_batch_count);
    Serial.println(" batched telemetry snapshots over MQTT...");

    JsonDocument bdoc;
    bdoc["machine_id"] = MACHINE_ID;
    bdoc["batch_count"] = telemetry_batch_count;
    JsonArray arr = bdoc["samples"].to<JsonArray>();

    for (int i = 0; i < telemetry_batch_count; i++) {
      JsonObject obj = arr.add<JsonObject>();
      obj["s"] = telemetry_batch[i].state;
      obj["b"] = round(telemetry_batch[i].baseline * 10.0f) / 10.0f;
      obj["p"] = telemetry_batch[i].peak;
      obj["u"] = telemetry_batch[i].uptime;
    }

    String topic = "factory/" + String(MACHINE_ID) + "/telemetry/batch";
    String jsonStr;
    serializeJson(bdoc, jsonStr);

    if (client.publish(topic.c_str(), jsonStr.c_str())) {
      Serial.println("[POWER] Batch publish SUCCESS!");
      telemetry_batch_count = 0;
      last_batch_flush_ms = millis();
    } else {
      Serial.println("[POWER] Batch publish FAILED, retaining buffer");
    }

    // Process any pending incoming MQTT commands (e.g. OTA update triggers)
    for (int i = 0; i < 5; i++) {
      client.loop();
      delay(50);
    }
  }

  disconnect_wifi_and_sleep();
}

void record_telemetry_snapshot(const char* state_str, float baseline, int peak, int uptime_sec) {
  if (telemetry_batch_count < BATCH_BUFFER_MAX) {
    strncpy(telemetry_batch[telemetry_batch_count].state, state_str, sizeof(telemetry_batch[0].state) - 1);
    telemetry_batch[telemetry_batch_count].state[sizeof(telemetry_batch[0].state) - 1] = '\0';
    telemetry_batch[telemetry_batch_count].baseline = baseline;
    telemetry_batch[telemetry_batch_count].peak = peak;
    telemetry_batch[telemetry_batch_count].uptime = uptime_sec;
    telemetry_batch_count++;
  } else {
    // Shift buffer if max reached before flush
    for (int i = 1; i < BATCH_BUFFER_MAX; i++) {
      telemetry_batch[i - 1] = telemetry_batch[i];
    }
    strncpy(telemetry_batch[BATCH_BUFFER_MAX - 1].state, state_str, sizeof(telemetry_batch[0].state) - 1);
    telemetry_batch[BATCH_BUFFER_MAX - 1].state[sizeof(telemetry_batch[0].state) - 1] = '\0';
    telemetry_batch[BATCH_BUFFER_MAX - 1].baseline = baseline;
    telemetry_batch[BATCH_BUFFER_MAX - 1].peak = peak;
    telemetry_batch[BATCH_BUFFER_MAX - 1].uptime = uptime_sec;
  }
}

void loop() {
  if (client.connected()) {
    client.loop();
  }

  int current_val = latest_audio_val;

  static unsigned long last_print_ms = 0;
  if (millis() - last_print_ms > 1000) {
    Serial.print("Audio level: ");
    Serial.print(current_val);
    Serial.print(" | Buffered snapshots: ");
    Serial.print(telemetry_batch_count);
    Serial.print("/60 | Wi-Fi: ");
    Serial.println(WiFi.status() == WL_CONNECTED ? "ON" : "SLEEP");
    last_print_ms = millis();
  }

  static float running_baseline = 0.0f;
  static float ambient_noise_floor = 0.0f;
  static bool machine_is_running = false;
  static unsigned long state_change_candidate_ms = 0;
  static unsigned long last_anomaly_recording_ms = 0;
  static unsigned long last_snapshot_record_ms = 0;

  // 1. Fast EMA for Live Volume Tracking
  if (running_baseline == 0.0f) {
    running_baseline = (float)current_val;
    ambient_noise_floor = min(380.0f, (float)current_val);
  } else {
    running_baseline = 0.90f * running_baseline + 0.10f * (float)current_val;
  }

  // 2. Self-Calibrating Ambient Noise Floor (Tracks lowest steady room noise)
  if ((float)current_val < ambient_noise_floor) {
    ambient_noise_floor = 0.85f * ambient_noise_floor + 0.15f * (float)current_val;
  } else {
    ambient_noise_floor = min(450.0f, 0.9995f * ambient_noise_floor + 0.0005f * (float)current_val);
  }

  // 3. Operational State Determination with Hysteresis
  float on_threshold = ambient_noise_floor + max(200.0f, ambient_noise_floor * 0.40f);
  float off_threshold = ambient_noise_floor + max(100.0f, ambient_noise_floor * 0.20f);

  if (!machine_is_running) {
    // Currently IDLE: Check if machine is turning ON
    if (running_baseline > on_threshold) {
      if (state_change_candidate_ms == 0) {
        state_change_candidate_ms = millis();
      } else if (millis() - state_change_candidate_ms > 300) { // Sustained for 300ms
        machine_is_running = true;
        state_change_candidate_ms = 0;
        Serial.println(">>> [STATE] Machine STARTUP Detected -> RUNNING <<<");

        // Trigger Instant Anomaly Audio Recording & Immediate Wakeup (5.0s)
        if (millis() - last_anomaly_recording_ms > 15000) {
          last_anomaly_recording_ms = millis();
          if (mp4.getRecordingState() != 1) {
            mp4.begin();
            Serial.println("Recording startup audio (5s)...");
          }

          // Immediate Wakeup & Publish Startup Alert
          if (ensure_network_connected()) {
            JsonDocument doc;
            doc["machine_id"] = MACHINE_ID;
            doc["duration_seconds"] = 5.0f;
            doc["peak_amplitude"] = current_val;
            char buffer[256];
            serializeJson(doc, buffer);
            client.publish(MQTT_TOPIC, buffer);
          }
        }
      }
    } else {
      state_change_candidate_ms = 0;
    }
  } else {
    // Currently RUNNING: Check if machine turned OFF
    if (running_baseline < off_threshold) {
      if (state_change_candidate_ms == 0) {
        state_change_candidate_ms = millis();
      } else if (millis() - state_change_candidate_ms > 3000) { // Must stay low for 3s
        machine_is_running = false;
        state_change_candidate_ms = 0;
        Serial.println(">>> [STATE] Machine SHUTDOWN / SPIN-DOWN Detected -> IDLE <<<");

        // Trigger Instant Shutdown / Spin-down Audio Recording (5.0s)
        if (millis() - last_anomaly_recording_ms > 10000) {
          last_anomaly_recording_ms = millis();
          if (mp4.getRecordingState() != 1) {
            mp4.begin();
            Serial.println("Recording shutdown / spin-down audio (5s)...");
          }

          // Immediate Wakeup & Publish Shutdown Alert
          if (ensure_network_connected()) {
            JsonDocument doc;
            doc["machine_id"] = MACHINE_ID;
            doc["duration_seconds"] = 5.0f;
            doc["peak_amplitude"] = current_val;
            char buffer[256];
            serializeJson(doc, buffer);
            client.publish(MQTT_TOPIC, buffer);
          }
        }
      }
    } else {
      state_change_candidate_ms = 0;

      // In-Run Abnormal Spike Detection: Sudden surge above steady running baseline
      if ((current_val > running_baseline + 800.0f || current_val > running_baseline * 1.8f) &&
          (millis() - last_anomaly_recording_ms > 20000)) {
        if (!latest_is_voice) {
          last_anomaly_recording_ms = millis();
          Serial.print(">>> [ANOMALY] In-Run Sound Spike! Level: ");
          Serial.print(current_val);
          Serial.print(" vs Baseline: ");
          Serial.println(running_baseline);

          if (mp4.getRecordingState() != 1) {
            mp4.begin();
          }

          // Immediate Wakeup & Publish Anomaly Alert
          if (ensure_network_connected()) {
            JsonDocument doc;
            doc["machine_id"] = MACHINE_ID;
            doc["duration_seconds"] = 5.0f;
            doc["peak_amplitude"] = current_val;
            char buffer[256];
            serializeJson(doc, buffer);
            client.publish(MQTT_TOPIC, buffer);
          }
        }
      }
    }
  }

  // 4. Record Internal Telemetry Snapshot to RAM every 30 seconds
  if (millis() - last_snapshot_record_ms > 30000) {
    last_snapshot_record_ms = millis();
    record_telemetry_snapshot(
      machine_is_running ? "RUNNING" : "IDLE",
      running_baseline,
      current_val,
      (int)(millis() / 1000)
    );
    Serial.print("[BUFFER] Logged 30s snapshot to RAM: State=");
    Serial.print(machine_is_running ? "RUNNING" : "IDLE");
    Serial.print(" | Baseline=");
    Serial.print(running_baseline);
    Serial.print(" | Total Snapshots=");
    Serial.println(telemetry_batch_count);
  }

  // 5. Scheduled 5-Minute Telemetry Batch Flush
  if (millis() - last_batch_flush_ms > BATCH_FLUSH_INTERVAL_MS || telemetry_batch_count >= BATCH_BUFFER_MAX) {
    flush_telemetry_batch_and_sleep();
  }

  // 6. Handle Completed MP4 Uploads and Return to Sleep
  recordingstate = (int)(mp4.getRecordingState());
  if (recordingstate == 0 && previousRecordingState == 1) {
      Serial.println("Recording finished, ensuring network and uploading MP4...");
      ensure_network_connected();
      encodeMP4andsendHttpPostRequest();
      // Flush any snapshots queued up to this anomaly and sleep
      flush_telemetry_batch_and_sleep();
  }
  previousRecordingState = recordingstate;

  delay(20);
}