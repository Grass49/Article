# main.py（更新版）
import json
import os
from orthogonal_table import ORTHOGONAL_L9
from atc_runner import run_atc
from metric_collector import collect_metrics
from analyzer import analyze, print_analysis
from npu_connector import NPUConnector
from accuracy_validator import validate_accuracy
from perf_monitor import PerfMonitor
from evaluator import evaluate, print_report, save_report
import infer_runner as infer_runner_mod
from infer_runner import run_infer

os.makedirs("results/om_models", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)

ONNX_MODEL  = input("输入ONNX模型路径: ")
input_shape = input("输入shape（如 input:1,3,224,224）: ")

npu_host = input("310B IP地址: ")
npu_user = input("310B用户名 [root]: ") or "root"
npu_pass = input("310B密码: ")
connector = NPUConnector(host=npu_host, username=npu_user, password=npu_pass)
connector.connect()

# 将connector注入infer_runner，使其使用真实msame推理而非随机数
infer_runner_mod.connector = connector

all_results = []

for i, cfg in enumerate(ORTHOGONAL_L9, start=1):
    print(f"\n{'='*55}")
    print(f"=== 实验 {i}/{len(ORTHOGONAL_L9)}  配置={cfg}")
    print(f"{'='*55}")

    # ── 1. ATC 转换 ───────────────────────────────────────────────────────────
    atc_result = run_atc(i, cfg, ONNX_MODEL, input_shape)
    if not atc_result["success"]:
        print(f"[跳过] Exp {i} ATC失败")
        continue
    om_path = atc_result["om_path"]

    # ── 2. 精度验证（ONNX vs OM 余弦相似度） ─────────────────────────────────
    cosine_result = {"cosine_mean": None, "cosine_min": None}
    try:
        cosine_result = validate_accuracy(
            onnx_path=ONNX_MODEL,
            om_path=om_path,
            input_shape=atc_result["input_shape_tuple"],
            connector=connector,
            num_samples=10,
        )
    except Exception as e:
        print(f"[警告] 精度验证失败: {e}")

    # ── 3. 推理性能测试 + 同步监控 ────────────────────────────────────────────
    monitor = PerfMonitor(connector, interval=0.5)
    monitor.start()

    try:
        latency, fps, infer_details = run_infer(
            om_path,
            atc_result["batch_size"],
            repeat=50,
            raw_log_path=f"results/logs/msame_exp{i}.log",
        )
    except Exception as e:
        print(f"[警告] 推理失败: {e}")
        monitor.stop(raw_log_path=f"results/logs/npu_smi_exp{i}.log")
        continue

    perf_stats = monitor.stop(raw_log_path=f"results/logs/npu_smi_exp{i}.log")

    # ── 4. 汇总 ───────────────────────────────────────────────────────────────
    metrics = collect_metrics(
        latency, fps, atc_result["batch_size"], infer_details
    )
    metrics["cosine_mean"] = cosine_result.get("cosine_mean")
    metrics["cosine_min"]  = cosine_result.get("cosine_min")
    metrics["accuracy_validation"] = cosine_result
    metrics["perf_stats"]  = perf_stats

    all_results.append({
        "exp_id": i,
        "config": cfg,
        "actual_input_shape": atc_result["actual_input_shape"],
        "atc_command": atc_result["command"],
        "metrics": metrics,
    })
    print(
        f"  → Batch={atc_result['batch_size']}  "
        f"shape={atc_result['actual_input_shape']}  FPS={fps:.1f}  "
        f"batch_latency={latency:.2f}ms  cosine={cosine_result.get('cosine_mean')}"
    )

connector.close()

# ── 5. 保存原始结果 ────────────────────────────────────────────────────────────
with open("results/results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print("\n[完成] 结果已写入 results/results.json")

# ── 6. 多指标正交分析 ──────────────────────────────────────────────────────────
analysis = analyze(all_results)
print_analysis(analysis)
with open("results/analysis.json", "w", encoding="utf-8") as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)

# ── 7. 综合评估 ────────────────────────────────────────────────────────────────
if all_results:
    eval_result = evaluate(all_results)
    print_report(eval_result)
    save_report(eval_result, "results/evaluation_report.json")
