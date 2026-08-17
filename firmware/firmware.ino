#include <ArduinoJson.h>
#include <PubSubClient.h>
#include <WiFi.h>
#include "AudioStream.h"
#include "StreamIO.h"
#include "AudioEncoder.h"
#include "MP4Recording.h"
#include "AmebaFatFS.h"
#include "Base64.h"

// Realtek SDK C API to disable the built-in auto-reconnect thread
extern "C" int wifi_config_autoreconnect(unsigned char, unsigned char, unsigned short);

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

const int RMS_THRESHOLD = 1500;
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

void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection as ");
    Serial.print(MACHINE_ID);
    Serial.print(" (user: ");
    Serial.print(MQTT_USER);
    Serial.print(")... ");
    if (client.connect(MACHINE_ID, MQTT_USER, MQTT_PASSWORD)) {
      Serial.println("connected!");
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
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  int current_val = latest_audio_val;

  static unsigned long last_print_ms = 0;
  if (millis() - last_print_ms > 500) {
    Serial.print("Audio level: ");
    Serial.print(current_val);
    Serial.print(" (Threshold: ");
    Serial.print(RMS_THRESHOLD);
    Serial.print(") | Anomaly state: ");
    Serial.println(is_anomalous ? "ACTIVE" : "IDLE");
    last_print_ms = millis();
  }

  static float running_baseline = 0.0f;
  static unsigned long last_delta_trigger_ms = 0;
  static unsigned long last_telemetry_heartbeat_ms = 0;

  // Periodic Telemetry Heartbeat (every 30 seconds)
  if (millis() - last_telemetry_heartbeat_ms > 30000) {
    last_telemetry_heartbeat_ms = millis();

    JsonDocument tdoc;
    tdoc["machine_id"] = MACHINE_ID;
    tdoc["machine_state"] = is_anomalous ? "RUNNING" : "IDLE";
    tdoc["running_baseline"] = running_baseline;
    tdoc["peak_amplitude"] = current_val;
    tdoc["uptime_seconds"] = (int)(millis() / 1000);

    char tbuf[256];
    serializeJson(tdoc, tbuf);

    String telemetryTopic = "factory/" + String(MACHINE_ID) + "/telemetry";
    client.publish(telemetryTopic.c_str(), tbuf);
    Serial.print("[HEARTBEAT] Published telemetry: ");
    Serial.println(tbuf);
  }

  if (current_val > RMS_THRESHOLD) {
    if (latest_is_voice && !is_anomalous) {
      // Human voice detected - suppress anomaly trigger
      static unsigned long last_voice_log_ms = 0;
      if (millis() - last_voice_log_ms > 1500) {
        Serial.println("[DSP] Human speech detected -> Anomaly suppressed");
        last_voice_log_ms = millis();
      }
    } else {
      last_loud_ms = millis();
      if (current_val > peak_val)
        peak_val = current_val;

      if (!is_anomalous) {
        // Machine startup or new discrete anomaly
        is_anomalous = true;
        anomaly_start_ms = millis();
        running_baseline = (float)current_val;
        Serial.println(">>> ANOMALY / MACHINE STARTUP DETECTED! <<<");
        if (mp4.getRecordingState() != 1) {
           mp4.begin();
           Serial.println("Recording audio started (5s)...");
        }
      } else {
        // Machine is already running: Smoothly track baseline hum
        running_baseline = 0.98f * running_baseline + 0.02f * (float)current_val;

        // In-Run Delta Detection: Sudden spike (+1200 or 1.8x baseline) above steady running sound
        if ((current_val > running_baseline + 1200.0f || current_val > running_baseline * 1.8f) && 
            (millis() - last_delta_trigger_ms > 12000)) {
          if (mp4.getRecordingState() != 1) {
            mp4.begin();
            last_delta_trigger_ms = millis();
            Serial.print(">>> IN-RUN SOUND DELTA! Level: ");
            Serial.print(current_val);
            Serial.print(" vs Running Baseline: ");
            Serial.println(running_baseline);
          }
        }
      }
    }
  }

  if (is_anomalous && (millis() - last_loud_ms > COOLDOWN_MS)) {
    float duration = (float)(last_loud_ms - anomaly_start_ms) / 1000.0f;
    if (duration < 0.1f) duration = 0.1f;
    is_anomalous = false;

    JsonDocument doc;
    doc["machine_id"] = MACHINE_ID;
    doc["duration_seconds"] = round(duration * 10.0f) / 10.0f;
    doc["peak_amplitude"] = peak_val;

    char buffer[256];
    serializeJson(doc, buffer);
    Serial.print("Publishing anomaly event: ");
    Serial.println(buffer);
    if (client.publish(MQTT_TOPIC, buffer)) {
      Serial.println("MQTT Publish SUCCESS");
    } else {
      Serial.println("MQTT Publish FAILED");
    }

    peak_val = 0;
    running_baseline = 0.0f;
  }

  recordingstate = (int)(mp4.getRecordingState());
  if (recordingstate == 0 && previousRecordingState == 1) {
      Serial.println("Recording finished, uploading...");
      encodeMP4andsendHttpPostRequest();
  }
  previousRecordingState = recordingstate;

  delay(20);
}