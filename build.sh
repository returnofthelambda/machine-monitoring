#!/usr/bin/env bash
set -e

if [ ! -f .env ]; then
  echo "[ERROR] .env file not found. Copy .env.example to .env and configure secrets."
  exit 1
fi

set -a
# shellcheck source=/dev/null
source .env
set +a

: "${BOARD_FQBN:="realtek:AmebaPro2:Ameba_AMB82-MINI:LoadNNModelSource=sd,OTAMode=Enable"}"
: "${PORT:="/dev/ttyUSB0"}"
: "${MQTT_USER:="factory_worker"}"
: "${UPLOAD_HOST:="192.168.1.100"}"
: "${UPLOAD_PORT:=5000}"

BUILD_FLAGS="-DWIFI_SSID=\"${WIFI_SSID}\" -DWIFI_PASSWORD=\"${WIFI_PASSWORD}\" -DMQTT_HOST=\"${MQTT_HOST}\" -DMQTT_PORT=${MQTT_PORT} -DUPLOAD_HOST=\"${UPLOAD_HOST}\" -DUPLOAD_PORT=${UPLOAD_PORT} -DMQTT_USER=\"${MQTT_USER}\" -DMQTT_PASSWORD=\"${MQTT_PASSWORD}\" -DMQTT_TOPIC=\"${MQTT_TOPIC}\" -DMACHINE_ID=\"${MACHINE_ID}\""

echo "==> Compiling firmware for ${MACHINE_ID} targeting ${MQTT_HOST}..."
arduino-cli compile \
  --fqbn "$BOARD_FQBN" \
  --build-property "build.extra_flags=-DARDUINO_SDK -DARDUINO_AMBPRO2 -DBOARD_AMB82_MINI {build.usb_flags} {build.menu_flags} $BUILD_FLAGS" \
  firmware/

if [[ "$*" == *"--flash"* ]]; then
  echo "==> Uploading binary to $PORT..."
  arduino-cli upload -p "$PORT" --fqbn "$BOARD_FQBN" firmware/
fi