# analyzer.py
from collections import defaultdict

# 修改说明：原版只对 fps 做主效应分析，引入余弦相似度后，
# 精度和速度的主效应 R 值分布往往完全不同——
# 例如 A 因素（精度模式）对余弦值的 R 最大，但对 FPS 未必是最显著因素。
# 改为支持多指标，让每个指标都有独立的因素影响力排名。

# 需要分析的指标列表，可按需增减
ANALYZE_METRICS = ["fps", "latency_ms", "cosine_mean"]

def _analyze_single_metric(results, metric_key):
    """对单个指标计算各因素的均值和极差 R。"""
    factors = ["A", "B", "C", "D"]
    stats = {f: defaultdict(list) for f in factors}

    for r in results:
        # 兼容 metric_key 不存在的情况（如精度验证未运行时 cosine_mean 为 None）
        val = r["metrics"].get(metric_key)
        if val is None:
            continue
        for f in factors:
            stats[f][r["config"][f]].append(val)

    analysis = {}
    for f in factors:
        if not stats[f]:
            continue
        means = {
            k: round(sum(v) / len(v), 4)
            for k, v in stats[f].items()
        }
        if len(means) < 2:
            R = 0.0
        else:
            R = round(max(means.values()) - min(means.values()), 4)
        analysis[f] = {"means": means, "R": R}

    return analysis


def analyze(results):
    """
    对 ANALYZE_METRICS 中的每个指标分别进行正交主效应分析。

    Returns:
        {
          "fps": {
            "A": {"means": {1: x, 2: y, 3: z}, "R": float},
            "B": ...,
            ...
            "factor_ranking": ["B", "A", "C", "D"]  # 按 R 值降序
          },
          "latency_ms": { ... },
          "cosine_mean": { ... },
        }
    """
    full_analysis = {}

    for metric in ANALYZE_METRICS:
        metric_analysis = _analyze_single_metric(results, metric)

        # 按 R 值降序排列因素，R 越大说明该因素对此指标影响越显著
        ranking = sorted(
            metric_analysis.keys(),
            key=lambda f: metric_analysis[f]["R"],
            reverse=True
        )
        metric_analysis["factor_ranking"] = ranking
        full_analysis[metric] = metric_analysis

    return full_analysis


def print_analysis(analysis):
    """打印格式化的分析报告。"""
    factor_labels = {
        "A": "精度模式 (A)",
        "B": "Batch Size (B)",
        "C": "算子实现模式 (C)",
        "D": "图优化策略 (D)",
    }

    for metric, data in analysis.items():
        print(f"\n{'='*50}")
        print(f"  指标：{metric}")
        print(f"{'='*50}")
        ranking = data.get("factor_ranking", [])
        print(f"  因素影响力排名（R值）：{' > '.join(ranking)}\n")
        for f in ranking:
            info = data[f]
            label = factor_labels.get(f, f)
            means_str = "  ".join(f"水平{k}={v}" for k, v in sorted(info["means"].items()))
            print(f"  {label}  R={info['R']}")
            print(f"    {means_str}")
