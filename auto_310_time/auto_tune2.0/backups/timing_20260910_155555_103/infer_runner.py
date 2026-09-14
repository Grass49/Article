# infer_runner.py
import os
import re
import shlex
from pathlib import Path

from npu_connector import NPUConnector

MSAME_BIN = "/home/HwHiAiUser/AscendProjects/tools/msame/out/msame"
SET_ENV = "export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64:$LD_LIBRARY_PATH"

connector = None


def _remote_file_size(path):
    quoted = shlex.quote(path)
    stdout, _, rc = connector.exec("stat -c %s -- {0} 2>/dev/null".format(quoted))
    if rc != 0:
        return None
    try:
        return int(stdout.strip())
    except (TypeError, ValueError):
        return None


def _ensure_remote_om(local_om, remote_om, force=False):
    local_size = os.path.getsize(local_om)
    remote_size = None if force else _remote_file_size(remote_om)

    if force or remote_size != local_size:
        if remote_size is None:
            reason = "远端文件不存在"
        elif force:
            reason = "强制重新上传"
        else:
            reason = "文件大小不一致(local={0}, remote={1})".format(local_size, remote_size)
        print("[infer] {0}，重新同步OM模型".format(reason))
        connector.upload(local_om, remote_om)
    else:
        print("[infer] 远端OM大小校验通过: {0} bytes".format(local_size))


def run_infer(om_path, batch_size, repeat=50, raw_log_path=None):
    if connector is None:
        raise RuntimeError("infer_runner.connector is not set")

    remote_base = "/home/HwHiAiUser/msame_test"
    local_om = om_path + ".om"
    remote_om = "{0}/{1}.om".format(remote_base, os.path.basename(om_path))
    remote_out = "{0}/output".format(remote_base)

    connector.exec("mkdir -p {0}/output".format(shlex.quote(remote_base)))
    # 不能只用test -f：SFTP中断可能留下一个存在但不完整的.om。
    _ensure_remote_om(local_om, remote_om)

    cmd = (
        "{env}; {msame} --model {model} --output {out} "
        "--outfmt TXT --loop {repeat} 2>&1"
    ).format(
        env=SET_ENV,
        msame=shlex.quote(MSAME_BIN),
        model=shlex.quote(remote_om),
        out=shlex.quote(remote_out),
        repeat=repeat,
    )

    def execute_once():
        connector.exec("rm -rf {0} && mkdir -p {0}".format(shlex.quote(remote_out)))
        return connector.exec(cmd, timeout=120)

    stdout, stderr, rc = execute_once()

    # 兼容本次修复前遗留的同名损坏OM；若msame明确报告模型文件无效，
    # 强制重新上传一次，再执行一次推理。
    error_text = (stdout or "") + "\n" + (stderr or "")
    invalid_om = (
        "invalid om file" in error_text.lower()
        or "load model from file failed" in error_text.lower()
    )
    if rc != 0 and invalid_om:
        print("[infer] 检测到远端OM无效，强制重新上传后重试一次...")
        _ensure_remote_om(local_om, remote_om, force=True)
        stdout, stderr, rc = execute_once()

    if raw_log_path:
        log_file = Path(raw_log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(stdout, encoding="utf-8")

    if rc != 0:
        detail = stdout.strip() or stderr.strip() or "<无错误输出>"
        raise RuntimeError("msame inference failed, returncode={0}\n{1}".format(rc, detail))

    latency = _parse_msame_latency(stdout)
    fps = (1000.0 / latency) * batch_size
    per_run = _parse_msame_per_run(stdout)
    latency_p95 = _percentile(per_run, 95) if per_run else latency
    latency_p99 = _percentile(per_run, 99) if per_run else latency

    if len(per_run) > 1:
        avg = sum(per_run) / len(per_run)
        std = (sum((x - avg) ** 2 for x in per_run) / len(per_run)) ** 0.5
        print(
            "  [infer] {0} loops: latency={1:.2f}ms fps={2:.1f} std={3:.2f} p95={4:.2f} p99={5:.2f}".format(
                repeat, latency, fps, std, latency_p95, latency_p99
            )
        )
    else:
        print("  [infer] latency={0:.2f}ms fps={1:.1f}".format(latency, fps))

    details = {
        "actual_batch_size": batch_size,
        "repeat": repeat,
        "per_run_latency_ms": per_run,
        "latency_p95_ms": round(latency_p95, 6),
        "latency_p99_ms": round(latency_p99, 6),
        "raw_log_path": raw_log_path,
    }
    return latency, fps, details


def _parse_msame_latency(output):
    patterns = [
        r"Inference average time without first time\s*[:：]\s*([\d.]+)\s*ms",
        r"Inference average time\s*[:：]\s*([\d.]+)\s*ms",
    ]
    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            return float(match.group(1))
    raise RuntimeError("Cannot parse msame average latency from output:\n{0}".format(output[:500]))


def _parse_msame_per_run(output):
    patterns = [
        r"Inference time\s*[:：]\s*([\d.]+)\s*ms",
        r"cost time\s*[:：]\s*([\d.]+)\s*ms",
    ]
    for pattern in patterns:
        values = [float(x) for x in re.findall(pattern, output, re.IGNORECASE)]
        if values:
            return values
    return []


def _percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 6)
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    value = ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    return round(value, 6)
