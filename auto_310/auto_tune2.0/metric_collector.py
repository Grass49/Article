# metric_collector.py


def collect_metrics(latency, fps, batch_size, infer_details=None):
    metrics = {
        "batch_latency_ms": round(latency, 6),
        "latency_ms": round(latency, 6),
        "actual_batch_size": int(batch_size),
        "equivalent_sample_latency_ms": round(latency / batch_size, 6),
        "fps": round(fps, 6),
    }
    if infer_details:
        metrics["inference_details"] = infer_details
        if infer_details.get("latency_p95_ms") is not None:
            metrics["latency_p95_ms"] = infer_details.get("latency_p95_ms")
        if infer_details.get("latency_p99_ms") is not None:
            metrics["latency_p99_ms"] = infer_details.get("latency_p99_ms")
    return metrics
