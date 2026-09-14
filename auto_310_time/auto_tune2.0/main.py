# main.py（更新版）
from config_timing import TimingJournal, install_interrupt_handlers
from config_execution import execute_configuration

import json
import os
from orthogonal_table import ORTHOGONAL_L9
from analyzer import analyze, print_analysis
from npu_connector import NPUConnector
from evaluator import evaluate, print_report, save_report
import infer_runner as infer_runner_mod

os.makedirs("results/om_models", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)

ONNX_MODEL  = input("输入ONNX模型路径: ")
input_shape = input("输入shape（如 input:1,3,224,224）: ")

npu_host = input("310B IP地址: ")
npu_user = input("310B用户名 [root]: ") or "root"
npu_pass = input("310B密码: ")
connector = NPUConnector(host=npu_host, username=npu_user, password=npu_pass)
install_interrupt_handlers()
connector.connect()

# 将connector注入infer_runner，使其使用真实msame推理而非随机数
infer_runner_mod.connector = connector

all_results = []

journal = TimingJournal()
print('[timing] output: {0}'.format(journal.directory))
try:
    for i, cfg in enumerate(ORTHOGONAL_L9, start=1):
        print('=== Experiment {0}/{1} config={2}'.format(i, len(ORTHOGONAL_L9), cfg))
        execute_configuration(
            connector, journal, i, cfg, ONNX_MODEL, input_shape, 50, all_results,
        )
finally:
    connector.close()

# ── 5. 保存原始结果 ────────────────────────────────────────────────────────────
with open("results/results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print("\n[完成] 结果已写入 results/results.json")

# ── 6. 多指标正交分析 ──────────────────────────────────────────────────────────
valid_results = [r for r in all_results if r.get('compile_success') and r.get('metrics')]
analysis = analyze(valid_results)
print_analysis(analysis)
with open("results/analysis.json", "w", encoding="utf-8") as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)

# ── 7. 综合评估 ────────────────────────────────────────────────────────────────
if valid_results:
    eval_result = evaluate(valid_results)
    print_report(eval_result)
    save_report(eval_result, "results/evaluation_report.json")
