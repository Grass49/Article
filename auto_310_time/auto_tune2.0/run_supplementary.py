#!/usr/bin/env python3
"""Run targeted supplementary experiments without overwriting the L9 results.

This script reuses the existing auto_tune2.0 modules, but writes all outputs
under results/supplementary_* so the original orthogonal experiment artifacts
remain intact.
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

from accuracy_validator import validate_accuracy
from atc_runner import run_atc
from evaluator import evaluate, save_report
from infer_runner import run_infer
import infer_runner as infer_runner_mod
from metric_collector import collect_metrics
from npu_connector import NPUConnector


TARGETED_CONFIGS = [
    {"name": "T1", "config": {"A": 3, "B": 3, "C": 1, "D": 1}},
    {"name": "T2", "config": {"A": 3, "B": 3, "C": 1, "D": 2}},
    {"name": "T3", "config": {"A": 3, "B": 3, "C": 2, "D": 2}},
    {"name": "T4", "config": {"A": 3, "B": 3, "C": 3, "D": 1}},
    {"name": "T5", "config": {"A": 3, "B": 3, "C": 3, "D": 2}},
]

STABILITY_CONFIGS = [
    {"source_exp": "Exp3", "config": {"A": 1, "B": 3, "C": 3, "D": 3}},
    {"source_exp": "Exp8", "config": {"A": 3, "B": 2, "C": 1, "D": 1}},
    {"source_exp": "Exp9", "config": {"A": 3, "B": 3, "C": 2, "D": 1}},
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--shape", required=True, help="Example: input:1,3,224,224")
    parser.add_argument("--npu_host", required=True)
    parser.add_argument("--npu_port", type=int, default=22)
    parser.add_argument("--npu_user", default="root")
    parser.add_argument("--npu_pass", default="")
    parser.add_argument("--mode", choices=["targeted", "stability", "all"], default="all")
    parser.add_argument("--infer_repeat", type=int, default=50)
    parser.add_argument("--stability_rounds", type=int, default=10)
    parser.add_argument("--accuracy_samples", type=int, default=10)
    return parser.parse_args()


def output_path(name):
    path = Path("results") / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_one(connector, args, exp_id, cfg, accuracy_samples):
    atc_result = run_atc(exp_id, cfg, args.onnx, args.shape)
    if not atc_result["success"]:
        return {
            "exp_id": exp_id,
            "config": cfg,
            "status": "atc_failed",
            "error": atc_result.get("error"),
        }

    om_path = atc_result["om_path"]
    input_shape_tuple = atc_result["input_shape_tuple"]

    cosine_result = {"cosine_mean": None, "cosine_min": None}
    try:
        cosine_result = validate_accuracy(
            onnx_path=args.onnx,
            om_path=om_path,
            input_shape=input_shape_tuple,
            connector=connector,
            num_samples=accuracy_samples,
        )
    except Exception as exc:
        cosine_result = {"cosine_mean": None, "cosine_min": None, "error": str(exc)}

    raw_dir = output_path("supplementary_raw_logs")
    latency, fps, infer_details = run_infer(
        om_path,
        atc_result["batch_size"],
        repeat=args.infer_repeat,
        raw_log_path=str(raw_dir / f"msame_exp{exp_id}.log"),
    )
    metrics = collect_metrics(
        latency, fps, atc_result["batch_size"], infer_details
    )
    metrics["cosine_mean"] = cosine_result.get("cosine_mean")
    metrics["cosine_min"] = cosine_result.get("cosine_min")
    metrics["accuracy_validation"] = cosine_result
    return {
        "exp_id": exp_id,
        "config": cfg,
        "actual_input_shape": atc_result["actual_input_shape"],
        "atc_command": atc_result["command"],
        "metrics": metrics,
    }


def run_targeted(connector, args):
    results = []
    for offset, item in enumerate(TARGETED_CONFIGS, start=101):
        print(f"\n[targeted] {item['name']} exp_id={offset} config={item['config']}")
        result = run_one(connector, args, offset, item["config"], args.accuracy_samples)
        if result:
            result["supplementary_name"] = item["name"]
            results.append(result)

    out_dir = output_path("supplementary_targeted")
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    successful = [r for r in results if "metrics" in r]
    if successful:
        eval_result = evaluate(successful)
        save_report(eval_result, str(out_dir / "evaluation_report.json"))
    return results


def summarize_rounds(rounds):
    latencies = [r["latency_ms"] for r in rounds]
    fps_values = [r["fps"] for r in rounds]
    return {
        "rounds": len(rounds),
        "latency_ms_mean": round(statistics.mean(latencies), 6),
        "latency_ms_std": round(statistics.pstdev(latencies), 6) if len(latencies) > 1 else 0.0,
        "fps_mean": round(statistics.mean(fps_values), 6),
        "fps_std": round(statistics.pstdev(fps_values), 6) if len(fps_values) > 1 else 0.0,
    }


def run_stability(connector, args):
    results = []
    for idx, item in enumerate(STABILITY_CONFIGS, start=201):
        print(f"\n[stability] {item['source_exp']} exp_id={idx} config={item['config']}")
        first = run_one(connector, args, idx, item["config"], args.accuracy_samples)
        if not first or "metrics" not in first:
            results.append(first or {"exp_id": idx, "config": item["config"], "status": "failed"})
            continue

        om_path = f"results/om_models/model_exp{idx}"
        rounds = [{
            "round": 1,
            "latency_ms": first["metrics"]["latency_ms"],
            "fps": first["metrics"]["fps"],
        }]
        actual_batch = first["metrics"]["actual_batch_size"]
        raw_dir = output_path("supplementary_raw_logs")
        for round_idx in range(2, args.stability_rounds + 1):
            latency, fps, infer_details = run_infer(
                om_path,
                actual_batch,
                repeat=args.infer_repeat,
                raw_log_path=str(
                    raw_dir / f"msame_{item['source_exp']}_round{round_idx}.log"
                ),
            )
            rounds.append({
                "round": round_idx,
                "latency_ms": latency,
                "fps": fps,
                "inference_details": infer_details,
            })

        entry = dict(first)
        entry["source_exp"] = item["source_exp"]
        entry["stability_rounds"] = rounds
        entry["stability_summary"] = summarize_rounds(rounds)
        results.append(entry)

    out_dir = output_path("supplementary_stability")
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def main():
    args = parse_args()
    connector = NPUConnector(
        host=args.npu_host,
        port=args.npu_port,
        username=args.npu_user,
        password=args.npu_pass,
    )
    connector.connect()
    infer_runner_mod.connector = connector

    try:
        if args.mode in ("targeted", "all"):
            run_targeted(connector, args)
        if args.mode in ("stability", "all"):
            run_stability(connector, args)
    finally:
        connector.close()


if __name__ == "__main__":
    main()
