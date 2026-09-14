#!/usr/bin/env python3
# run_pipeline.py
import argparse
import json
import os
import sys
from config_timing import TimingJournal, install_interrupt_handlers

os.chdir(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--onnx", required=True)
parser.add_argument("--shape", required=True, help="example: input:1,3,224,224")
parser.add_argument("--npu_host", required=True)
parser.add_argument("--npu_port", type=int, default=22)
parser.add_argument("--npu_user", default="root")
parser.add_argument("--npu_pass", default="")
parser.add_argument("--no_tune", action="store_true", help="run only one baseline config")
parser.add_argument("--l9_only", action="store_true", help="run only L9 initial experiments")
parser.add_argument("--repeat", type=int, default=50)
args = parser.parse_args()

from config_execution import execute_configuration
from analyzer import analyze, print_analysis
from evaluator import evaluate, print_report, save_report
from npu_connector import NPUConnector
from offline_analysis import BUDGETS, compare_methods
from orthogonal_table import ORTHOGONAL_L9
from param_space import FULL_FACTORIAL_81
import infer_runner as infer_runner_mod


def choose_experiments():
    if args.no_tune:
        return [{"A": 1, "B": 1, "C": 1, "D": 1}]
    if args.l9_only:
        return ORTHOGONAL_L9
    return FULL_FACTORIAL_81


def save_json(path, data):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


os.makedirs("results/om_models", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)

print("[pipeline] ONNX: {0}".format(args.onnx), flush=True)
print("[pipeline] shape: {0}".format(args.shape), flush=True)

connector = NPUConnector(
    host=args.npu_host,
    port=args.npu_port,
    username=args.npu_user,
    password=args.npu_pass,
)
install_interrupt_handlers()
connector.connect()
infer_runner_mod.connector = connector

experiments = choose_experiments()
all_results = []
journal = TimingJournal()
print('[timing] output: {0}'.format(journal.directory), flush=True)

try:
    for i, cfg in enumerate(experiments, start=1):
        total = len(experiments)
        print("\n{0}".format("=" * 55), flush=True)
        print("=== Experiment {0}/{1} config={2}".format(i, total, cfg), flush=True)
        print("{0}".format("=" * 55), flush=True)

        execute_configuration(
            connector, journal, i, cfg, args.onnx, args.shape,
            args.repeat, all_results,
        )

finally:
    connector.close()

save_json("results/results.json", all_results)
print("\n[done] raw results saved to results/results.json", flush=True)

valid_results = [r for r in all_results if r.get("compile_success") and r.get("metrics")]
if valid_results:
    analysis = analyze(valid_results)
    print_analysis(analysis)
    save_json("results/analysis.json", analysis)

    eval_result = evaluate(valid_results)
    print_report(eval_result)
    save_report(eval_result, "results/evaluation_report.json")

if len(valid_results) >= 9:
    by_config = {}
    for row in valid_results:
        cfg = row.get("config")
        if cfg:
            by_config[(cfg["A"], cfg["B"], cfg["C"], cfg["D"])] = row
    offline_report = compare_methods(by_config, BUDGETS, random_trials=30)
    save_json("results/offline_comparison.json", offline_report)
    print("[done] offline comparison saved to results/offline_comparison.json", flush=True)
