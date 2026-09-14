# evaluator.py
# 约束优先的部署评估：余弦相似度作为合格约束，只对真实可用且有区分度的指标评分。

import json
import os


COSINE_THRESHOLD = 0.999

# 资源指标缺失时会自动退出评分，并对剩余可用权重重新归一化。
OBJECTIVE_SPECS = {
    "fps": {
        "higher_is_better": True,
        "path": ("metrics", "fps"),
    },
    "temperature": {
        "higher_is_better": False,
        "path": ("metrics", "perf_stats", "die_temp_avg"),
    },
    "memory_mb": {
        "higher_is_better": False,
        "path": ("metrics", "perf_stats", "memory_used_mb_avg"),
    },
}


def _get_path(item, path):
    value = item
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def normalize(values, higher_is_better=True):
    """Min-max normalize a metric that is known to contain real variation."""
    minimum, maximum = min(values), max(values)
    if maximum == minimum:
        raise ValueError("没有区分度的指标不应进入综合评分")
    if higher_is_better:
        return [(value - minimum) / (maximum - minimum) for value in values]
    return [(maximum - value) / (maximum - minimum) for value in values]


def compute_scores(results):
    if not results:
        return [], {}

    active = {}
    normalized = {}
    for name, spec in OBJECTIVE_SPECS.items():
        values = [_get_path(result, spec["path"]) for result in results]
        # 不用默认值冒充实测值；必须每组都有数据且存在真实差异。
        if any(value is None for value in values) or len(set(values)) < 2:
            continue
        active[name] = spec
        normalized[name] = normalize(values, spec["higher_is_better"])

    # No scenario-specific preference was supplied, so use a transparent
    # equal-weight baseline over only the measured, discriminative objectives.
    equal_weight = 1.0 / len(active) if active else 0.0
    weights_used = {
        name: round(equal_weight, 6) for name in active
    }

    scored = []
    for index, result in enumerate(results):
        cosine = _get_path(result, ("metrics", "cosine_mean"))
        feasible = cosine is not None and cosine >= COSINE_THRESHOLD
        if feasible and weights_used:
            score = sum(
                weights_used[name] * normalized[name][index]
                for name in weights_used
            )
        elif feasible:
            score = 0.0
        else:
            score = -1.0

        entry = dict(result)
        entry["score"] = round(score, 6)
        entry["feasible"] = feasible
        entry["constraint"] = {
            "cosine_threshold": COSINE_THRESHOLD,
            "cosine_observed": cosine,
        }
        scored.append(entry)

    return scored, weights_used


def evaluate(results):
    if not results:
        raise ValueError("results列表为空，无法评估")

    scored, weights_used = compute_scores(results)
    feasible = [result for result in scored if result["feasible"]]
    ranking = sorted(scored, key=lambda item: item["score"], reverse=True)

    fastest = max(scored, key=lambda item: item["metrics"]["fps"])
    lowest_latency = min(scored, key=lambda item: item["metrics"]["latency_ms"])
    most_accurate = max(
        scored,
        key=lambda item: item["metrics"].get("cosine_mean")
        if item["metrics"].get("cosine_mean") is not None else float("-inf"),
    )
    recommended = max(feasible, key=lambda item: item["score"]) if feasible else None

    for result in ranking:
        result["tags"] = []
        if recommended and result["exp_id"] == recommended["exp_id"]:
            result["tags"].append("⭐ 约束内推荐")
        if result["exp_id"] == fastest["exp_id"]:
            result["tags"].append("🚀 最高吞吐")
        if result["exp_id"] == lowest_latency["exp_id"]:
            result["tags"].append("⏱ 最低批时延")
        if result["exp_id"] == most_accurate["exp_id"]:
            result["tags"].append("🎯 一致性最高")
        if not result["feasible"]:
            result["tags"].append("⚠ 未通过一致性约束")

    return {
        "cosine_threshold": COSINE_THRESHOLD,
        "weights_used": weights_used,
        "feasible_count": len(feasible),
        "fastest": fastest,
        "lowest_latency": lowest_latency,
        "most_accurate": most_accurate,
        "recommended": recommended,
        "ranking": ranking,
    }


def print_report(eval_result):
    print("\n" + "=" * 60)
    print("  昇腾310B 模型评测报告（余弦约束优先）")
    print("=" * 60)
    print(f"  余弦相似度阈值: {eval_result['cosine_threshold']}")
    print(f"  实际参与评分的权重: {eval_result['weights_used']}")
    print(f"  通过约束: {eval_result['feasible_count']}/{len(eval_result['ranking'])}")

    for label, key in [
        ("⭐ 约束内推荐", "recommended"),
        ("🚀 最高吞吐", "fastest"),
        ("⏱ 最低批时延", "lowest_latency"),
        ("🎯 一致性最高", "most_accurate"),
    ]:
        result = eval_result[key]
        if result is None:
            print(f"\n{label}: 无符合条件的配置")
            continue
        metrics = result["metrics"]
        print(f"\n{label}: Exp #{result['exp_id']}")
        print(f"  配置: {result['config']}")
        print(
            f"  Batch={metrics.get('actual_batch_size')}  "
            f"批时延={metrics.get('latency_ms'):.4f}ms  "
            f"FPS={metrics.get('fps'):.2f}  "
            f"余弦={metrics.get('cosine_mean')}  得分={result['score']}"
        )

    print("\n📊 完整排名:")
    for index, result in enumerate(eval_result["ranking"], 1):
        metrics = result["metrics"]
        tags = " ".join(result.get("tags", []))
        print(
            f"  #{index:<2} Exp {result['exp_id']:<4} "
            f"B={metrics.get('actual_batch_size')!s:<3} "
            f"FPS={metrics.get('fps'):<12.2f} "
            f"latency={metrics.get('latency_ms'):<9.4f} "
            f"score={result['score']:<8.4f} {tags}"
        )
    print("=" * 60)


def _strip_private_fields(result):
    if result is None:
        return None
    return {key: value for key, value in result.items() if not key.startswith("_")}


def save_report(eval_result, path="results/evaluation_report.json"):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    clean = {
        "cosine_threshold": eval_result["cosine_threshold"],
        "weights_used": eval_result["weights_used"],
        "feasible_count": eval_result["feasible_count"],
        "fastest": _strip_private_fields(eval_result["fastest"]),
        "lowest_latency": _strip_private_fields(eval_result["lowest_latency"]),
        "most_accurate": _strip_private_fields(eval_result["most_accurate"]),
        "recommended": _strip_private_fields(eval_result["recommended"]),
        "ranking": [_strip_private_fields(result) for result in eval_result["ranking"]],
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(clean, file, indent=2, ensure_ascii=False)
    print(f"[评估] 报告已保存到 {path}")
