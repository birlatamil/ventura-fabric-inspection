"""
mqtt_output.py
==============
MQTT publisher for fabric width measurements.

Publishes to two topics:
  fabric/width   — measurement JSON (5 Hz, QoS 1, retained)
  fabric/health  — system health heartbeat (every 5 s, QoS 0)

JSON payload (fabric/width)
----------------------------
{
  "timestamp":  "2026-02-21T12:33:55+05:30",
  "epoch":      1708503235.123,
  "width_mm":   1245.7,
  "left_mm":    102.3,
  "right_mm":   1348.0,
  "confidence": 0.97,
  "num_lines":  38,
  "jitter_mm":  0.12,
  "status":     "OK"   // "OK" | "LOW_CONFIDENCE" | "OUT_OF_RANGE"
}

Reconnects automatically if the broker goes away.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import paho.mqtt.client as mqtt

log = logging.getLogger(__name__)

# ── Timezone helper (IST = UTC+5:30) ───────────────────────────────────────
_IST = timezone(timedelta(hours=5, minutes=30))


def _now_iso() -> str:
    return datetime.now(_IST).isoformat(timespec="seconds")


# ── Publisher class ─────────────────────────────────────────────────────────

class MQTTPublisher:
    """
    Thread-safe MQTT publisher with auto-reconnect.

    Parameters
    ----------
    broker : str
        MQTT broker hostname or IP.
    port : int
        Broker port (default 1883, TLS usually 8883).
    topic_width : str
        Topic for width measurement messages.
    topic_health : str
        Topic for health heartbeat messages.
    client_id : str
        MQTT client identifier (unique per device).
    keepalive : int
        MQTT keepalive interval in seconds.
    qos : int
        Quality of service level (0, 1, or 2).
    retain : bool
        Retain the last message on the broker.
    reconnect_delay_s : float
        Seconds to wait between reconnect attempts.
    username, password : str
        MQTT credentials (leave empty for anonymous).
    """

    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        topic_width: str = "fabric/width",
        topic_health: str = "fabric/health",
        client_id: str = "fabric_width_system",
        keepalive: int = 60,
        qos: int = 1,
        retain: bool = True,
        reconnect_delay_s: float = 5.0,
        username: str = "",
        password: str = "",
    ) -> None:
        self._broker = broker
        self._port = port
        self._topic_width = topic_width
        self._topic_health = topic_health
        self._qos = qos
        self._retain = retain
        self._reconnect_delay = reconnect_delay_s

        self._client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)
        if username:
            self._client.username_pw_set(username, password)

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_publish = self._on_publish

        # Stats
        self._published_count: int = 0
        self._connect_count: int = 0
        self._connected: bool = False
        self._lock = threading.Lock()

        # Keepalive
        self._keepalive = keepalive

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Connect to the broker and start the background network loop."""
        try:
            self._client.connect(self._broker, self._port, self._keepalive)
            self._client.loop_start()
            log.info("MQTT connecting to %s:%d …", self._broker, self._port)
        except Exception as exc:
            log.error("MQTT initial connect failed: %s — will retry automatically.", exc)
            self._client.loop_start()

    def disconnect(self) -> None:
        """Gracefully disconnect."""
        self._client.loop_stop()
        self._client.disconnect()
        log.info("MQTT disconnected. published=%d connects=%d",
                 self._published_count, self._connect_count)

    # ── Publishing ─────────────────────────────────────────────────────────

    def publish_width(
        self,
        width_mm: float,
        left_mm: float,
        right_mm: float,
        confidence: float,
        num_lines: int,
        jitter_mm: Optional[float] = None,
    ) -> bool:
        """
        Publish a width measurement.  Returns True if the publish was queued.
        """
        status = self._status_string(width_mm, confidence)
        payload = self._build_payload(
            width_mm=width_mm,
            left_mm=left_mm,
            right_mm=right_mm,
            confidence=confidence,
            num_lines=num_lines,
            jitter_mm=jitter_mm,
            status=status,
        )
        return self._publish(self._topic_width, payload, qos=self._qos, retain=self._retain)

    def publish_health(self, health_dict: dict) -> bool:
        """Publish a system health update (low QoS, not retained)."""
        payload = json.dumps(health_dict, ensure_ascii=False)
        return self._publish(self._topic_health, payload, qos=0, retain=False)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _build_payload(
        self,
        width_mm: float,
        left_mm: float,
        right_mm: float,
        confidence: float,
        num_lines: int,
        jitter_mm: Optional[float] = None,
        status: str = "OK",
    ) -> str:
        payload: dict = {
            "timestamp": _now_iso(),
            "epoch": round(time.time(), 3),
            "width_mm": round(width_mm, 3),
            "left_mm": round(left_mm, 3),
            "right_mm": round(right_mm, 3),
            "confidence": round(confidence, 4),
            "num_lines": num_lines,
            "status": status,
        }
        if jitter_mm is not None:
            payload["jitter_mm"] = round(jitter_mm, 4)
        return json.dumps(payload, ensure_ascii=False)

    def _status_string(self, width_mm: float, confidence: float) -> str:
        if confidence < 0.6:
            return "LOW_CONFIDENCE"
        if not (800.0 <= width_mm <= 1900.0):
            return "OUT_OF_RANGE"
        return "OK"

    def _publish(self, topic: str, payload: str, qos: int, retain: bool) -> bool:
        try:
            info = self._client.publish(topic, payload, qos=qos, retain=retain)
            if info.rc == mqtt.MQTT_ERR_SUCCESS:
                with self._lock:
                    self._published_count += 1
                return True
            log.warning("MQTT publish returned rc=%d on topic=%s", info.rc, topic)
            return False
        except Exception as exc:
            log.error("MQTT publish exception: %s", exc)
            return False

    # ── MQTT callbacks ─────────────────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            self._connected = True
            self._connect_count += 1
            log.info("MQTT connected (rc=0, total_connects=%d)", self._connect_count)
        else:
            log.error("MQTT connection refused — rc=%d. Retrying …", rc)
            threading.Timer(self._reconnect_delay, self._retry_connect).start()

    def _on_disconnect(self, client, userdata, rc) -> None:
        self._connected = False
        if rc != 0:
            log.warning("MQTT unexpectedly disconnected (rc=%d). Reconnecting …", rc)
            threading.Timer(self._reconnect_delay, self._retry_connect).start()

    def _on_publish(self, client, userdata, mid) -> None:
        log.debug("MQTT message mid=%d delivered.", mid)

    def _retry_connect(self) -> None:
        try:
            self._client.reconnect()
        except Exception as exc:
            log.error("MQTT reconnect failed: %s — will retry in %.0fs.", exc, self._reconnect_delay)
            threading.Timer(self._reconnect_delay, self._retry_connect).start()

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def published_count(self) -> int:
        with self._lock:
            return self._published_count

    @property
    def is_connected(self) -> bool:
        return self._connected
