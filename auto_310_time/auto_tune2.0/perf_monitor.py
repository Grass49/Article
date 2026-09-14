# perf_monitor.py
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from npu_connector import NPUConnector


@dataclass
class NPUSample:
    timestamp: float
    ai_core_util: Optional[float] = None
    ai_cpu_util: Optional[float] = None
    ctrl_cpu_util: Optional[float] = None
    die_temp: Optional[float] = None
    board_temp: Optional[float] = None
    memory_used_mb: Optional[float] = None
    memory_total_mb: Optional[float] = None
    power_w: Optional[float] = None


def _first_float(patterns, output):
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def parse_npu_smi(output):
    sample = NPUSample(timestamp=time.time())
    sample.ai_core_util = _first_float([r"AI Core\s*[:：]\s*([\d.]+)\s*%"], output)
    sample.ai_cpu_util = _first_float([r"AI CPU\s*[:：]\s*([\d.]+)\s*%"], output)
    sample.ctrl_cpu_util = _first_float([r"Ctrl CPU\s*[:：]\s*([\d.]+)\s*%"], output)
    sample.die_temp = _first_float([r"Die Temp\s*[:：]\s*([\d.]+)", r"Temperature\s*[:：]\s*([\d.]+)"], output)
    sample.board_temp = _first_float([r"Board Temp\s*[:：]\s*([\d.]+)"], output)
    sample.power_w = _first_float(
        [
            r"Power\s*[:：]\s*([\d.]+)\s*W",
            r"Power\(W\)\s*[:：]?\s*([\d.]+)",
            r"([0-9.]+)\s*W\s*/",
        ],
        output,
    )

    mem_match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*MiB", output)
    if mem_match:
        sample.memory_used_mb = float(mem_match.group(1))
        sample.memory_total_mb = float(mem_match.group(2))
    return sample


class PerfMonitor:
    def __init__(self, connector, device_id=0, interval=0.5):
        self.connector = connector
        self.device_id = device_id
        self.interval = interval
        self._samples = []
        self._raw_outputs = []
        self._stop_event = threading.Event()
        self._thread = None
        self._monitor_connector = None

    def _poll(self):
        while not self._stop_event.is_set():
            try:
                out, _, _ = self._monitor_connector.exec(
                    "npu-smi info -t usages -i {0}".format(self.device_id),
                    timeout=10,
                )
                sample = parse_npu_smi(out)
                self._samples.append(sample)
                self._raw_outputs.append((sample.timestamp, out))
            except Exception as exc:
                print("[PerfMonitor] sample failed: {0}".format(exc))
            time.sleep(self.interval)

    def start(self):
        self._samples = []
        self._raw_outputs = []
        self._stop_event.clear()
        self._monitor_connector = self.connector.clone()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        print("[PerfMonitor] started interval={0}s".format(self.interval))

    def stop(self, raw_log_path=None):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._monitor_connector:
            self._monitor_connector.close()
            self._monitor_connector = None
        stats = self._compute_stats()
        if raw_log_path:
            path = Path(raw_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            blocks = [
                "timestamp={0}\n{1}".format(timestamp, output.rstrip())
                for timestamp, output in self._raw_outputs
            ]
            path.write_text("\n\n===== SAMPLE =====\n\n".join(blocks), encoding="utf-8")
            stats["raw_log_path"] = raw_log_path
        print("[PerfMonitor] collected {0} samples".format(len(self._samples)))
        return stats

    def _compute_stats(self):
        if not self._samples:
            return {}

        def values(key):
            return [getattr(s, key) for s in self._samples if getattr(s, key) is not None]

        def avg(key):
            vals = values(key)
            return round(sum(vals) / len(vals), 2) if vals else None

        def peak(key):
            vals = values(key)
            return round(max(vals), 2) if vals else None

        def first(key):
            vals = values(key)
            return vals[0] if vals else None

        duration = self._samples[-1].timestamp - self._samples[0].timestamp if len(self._samples) > 1 else 0.0
        power_vals = values("power_w")
        avg_power = round(sum(power_vals) / len(power_vals), 2) if power_vals else None

        mem_start = first("memory_used_mb")
        mem_peak = peak("memory_used_mb")
        mem_delta = round(mem_peak - mem_start, 2) if mem_peak is not None and mem_start is not None else None
        energy_j = round(avg_power * duration, 6) if avg_power is not None else None

        return {
            "sample_count": len(self._samples),
            "duration_s": round(duration, 2),
            "ai_core_util_avg": avg("ai_core_util"),
            "ai_core_util_peak": peak("ai_core_util"),
            "ai_cpu_util_avg": avg("ai_cpu_util"),
            "ai_cpu_util_peak": peak("ai_cpu_util"),
            "ctrl_cpu_util_avg": avg("ctrl_cpu_util"),
            "ctrl_cpu_util_peak": peak("ctrl_cpu_util"),
            "die_temp_avg": avg("die_temp"),
            "die_temp_peak": peak("die_temp"),
            "board_temp_avg": avg("board_temp"),
            "board_temp_peak": peak("board_temp"),
            "memory_used_mb_avg": avg("memory_used_mb"),
            "memory_used_mb_peak": mem_peak,
            "memory_used_mb_start": mem_start,
            "memory_used_mb_delta_peak": mem_delta,
            "memory_total_mb": self._samples[-1].memory_total_mb,
            "power_w_avg": avg_power,
            "power_w_peak": peak("power_w"),
            "energy_j": energy_j,
        }
