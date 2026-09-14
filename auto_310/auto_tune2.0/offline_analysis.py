# offline_analysis.py
import argparse
import json
import math
import os
import random

from orthogonal_table import ORTHOGONAL_L9
from param_space import FACTOR_LEVELS, FULL_FACTORIAL_81, LEVEL_VALUES, decode_config

DEFAULT_BUDGET = 18
BUDGETS = [9, 12, 15, 18, 21, 27]
COSINE_THRESHOLD = 0.999

OBJECTIVES = [
    ("fps", ("metrics", "fps"), True),
    ("latency_ms", ("metrics", "latency_ms"), False),
    ("latency_p95_ms", ("metrics", "latency_p95_ms"), False),
    ("latency_p99_ms", ("metrics", "latency_p99_ms"), False),
    ("memory_delta_peak_mb", ("metrics", "perf_stats", "memory_used_mb_delta_peak"), False),
    ("memory_peak_mb", ("metrics", "perf_stats", "memory_used_mb_peak"), False),
    ("power_w", ("metrics", "perf_stats", "power_w_avg"), False),
    ("energy_j", ("metrics", "perf_stats", "energy_j"), False),
]


def config_key(cfg):
    return tuple(cfg[f] for f in FACTOR_LEVELS)


def get_path(row, path):
    value = row
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    by_config = {}
    for row in rows:
        cfg = row.get("config")
        if cfg:
            row.setdefault("decoded_config", decode_config(cfg))
            by_config[config_key(cfg)] = row
    return rows, by_config


def feasible_rows(rows):
    out = []
    for row in rows:
        cosine = get_path(row, ("metrics", "cosine_mean"))
        if cosine is None or cosine >= COSINE_THRESHOLD:
            out.append(row)
    return out


def active_objectives(rows):
    active = []
    for name, path, higher in OBJECTIVES:
        values = [get_path(row, path) for row in rows]
        if any(v is None for v in values):
            continue
        if len(set(values)) < 2:
            continue
        active.append((name, path, higher))
    return active


def normalized_matrix(rows, objectives):
    matrix = []
    columns = []
    for name, path, higher in objectives:
        values = [float(get_path(row, path)) for row in rows]
        lo = min(values)
        hi = max(values)
        if hi == lo:
            continue
        if higher:
            col = [(v - lo) / (hi - lo) for v in values]
        else:
            col = [(hi - v) / (hi - lo) for v in values]
        columns.append((name, col))
    for i in range(len(rows)):
        matrix.append([col[i] for _, col in columns])
    return [name for name, _ in columns], matrix


def critic_weights(rows):
    rows = feasible_rows(rows)
    objectives = active_objectives(rows)
    names, matrix = normalized_matrix(rows, objectives)
    if not rows or not names:
        return {}, []
    n = len(rows)
    m = len(names)
    means = [sum(matrix[i][j] for i in range(n)) / n for j in range(m)]
    stds = []
    for j in range(m):
        variance = sum((matrix[i][j] - means[j]) ** 2 for i in range(n)) / n
        stds.append(math.sqrt(variance))
    conflicts = []
    for j in range(m):
        conflict = 0.0
        for k in range(m):
            corr = _pearson([matrix[i][j] for i in range(n)], [matrix[i][k] for i in range(n)])
            conflict += 1.0 - corr
        conflicts.append(conflict)
    info = [stds[i] * conflicts[i] for i in range(m)]
    total = sum(info)
    if total <= 0:
        weight = 1.0 / m
        weights = dict((name, round(weight, 6)) for name in names)
    else:
        weights = dict((names[i], round(info[i] / total, 6)) for i in range(m))
    return weights, names


def _pearson(left, right):
    n = len(left)
    if n < 2:
        return 0.0
    lm = sum(left) / n
    rm = sum(right) / n
    num = sum((left[i] - lm) * (right[i] - rm) for i in range(n))
    ld = math.sqrt(sum((x - lm) ** 2 for x in left))
    rd = math.sqrt(sum((x - rm) ** 2 for x in right))
    if ld == 0 or rd == 0:
        return 0.0
    return max(-1.0, min(1.0, num / (ld * rd)))


def guidance_scores(rows, weights):
    objective_map = dict((name, (path, higher)) for name, path, higher in OBJECTIVES)
    objectives = [(name, objective_map[name][0], objective_map[name][1]) for name in weights if name in objective_map]
    names, matrix = normalized_matrix(rows, objectives)
    scores = {}
    for i, row in enumerate(rows):
        score = 0.0
        for j, name in enumerate(names):
            score += weights[name] * matrix[i][j]
        scores[config_key(row["config"])] = score
    return scores


def main_effect_responses(l9_rows, scores):
    responses = {}
    for factor in FACTOR_LEVELS:
        responses[factor] = {}
        for level in LEVEL_VALUES:
            vals = [scores[config_key(row["config"])] for row in l9_rows if row["config"][factor] == level]
            responses[factor][level] = sum(vals) / len(vals) if vals else 0.0
    return responses


def predict_config_score(cfg, responses):
    return sum(responses[f][cfg[f]] for f in FACTOR_LEVELS)


def proposed_selection(by_config, budget):
    l9_keys = set(config_key(cfg) for cfg in ORTHOGONAL_L9)
    l9_rows = [by_config[key] for key in l9_keys if key in by_config]
    weights, used_objectives = critic_weights(l9_rows)
    scores = guidance_scores(l9_rows, weights)
    responses = main_effect_responses(l9_rows, scores)
    candidates = []
    for cfg in FULL_FACTORIAL_81:
        key = config_key(cfg)
        if key in l9_keys:
            continue
        candidates.append(
            {
                "config": cfg,
                "predicted_guidance_score": round(predict_config_score(cfg, responses), 6),
            }
        )
    candidates.sort(key=lambda x: x["predicted_guidance_score"], reverse=True)
    selected_keys = list(l9_keys)
    selected_keys.extend(config_key(item["config"]) for item in candidates[: max(0, budget - 9)])
    selected_rows = [by_config[key] for key in selected_keys if key in by_config]
    return selected_rows, {
        "critic_weights": weights,
        "used_objectives": used_objectives,
        "main_effect_responses": responses,
        "candidate_priority": candidates,
    }


def topsis(rows, weights=None):
    rows = feasible_rows(rows)
    if not rows:
        return []
    if weights is None:
        weights, _ = critic_weights(rows)
    objectives = []
    for name, path, higher in OBJECTIVES:
        if name in weights:
            objectives.append((name, path, higher))
    names, matrix = normalized_matrix(rows, objectives)
    if not names:
        return []
    n = len(rows)
    m = len(names)
    weighted = []
    for i in range(n):
        weighted.append([matrix[i][j] * weights[names[j]] for j in range(m)])
    ideal = [max(weighted[i][j] for i in range(n)) for j in range(m)]
    nadir = [min(weighted[i][j] for i in range(n)) for j in range(m)]
    ranked = []
    for i, row in enumerate(rows):
        d_pos = math.sqrt(sum((weighted[i][j] - ideal[j]) ** 2 for j in range(m)))
        d_neg = math.sqrt(sum((weighted[i][j] - nadir[j]) ** 2 for j in range(m)))
        score = d_neg / (d_pos + d_neg) if d_pos + d_neg > 0 else 0.0
        item = dict(row)
        item["topsis_score"] = round(score, 6)
        ranked.append(item)
    ranked.sort(key=lambda x: x["topsis_score"], reverse=True)
    return ranked


def method_summary(rows, label, weights=None):
    ranking = topsis(rows, weights)
    best = ranking[0] if ranking else None
    return {
        "method": label,
        "evaluated_count": len(rows),
        "feasible_count": len(feasible_rows(rows)),
        "best": best,
        "ranking": ranking,
    }


def compare_methods(by_config, budgets, random_trials=30, seed=2026):
    all_rows = [by_config[config_key(cfg)] for cfg in FULL_FACTORIAL_81 if config_key(cfg) in by_config]
    full_weights, _ = critic_weights(all_rows)
    rng = random.Random(seed)
    report = {
        "cosine_threshold": COSINE_THRESHOLD,
        "budgets": {},
        "full_search": method_summary(all_rows, "Full Search", full_weights),
    }
    l9_rows = [by_config[config_key(cfg)] for cfg in ORTHOGONAL_L9 if config_key(cfg) in by_config]
    for budget in budgets:
        proposed_rows, proposed_meta = proposed_selection(by_config, budget)
        random_best_scores = []
        random_bests = []
        for _ in range(random_trials):
            sample = rng.sample(all_rows, min(budget, len(all_rows)))
            summary = method_summary(sample, "Random Search", full_weights)
            best = summary["best"]
            if best:
                random_best_scores.append(best["topsis_score"])
                random_bests.append(best)
        report["budgets"][str(budget)] = {
            "L9 Only": method_summary(l9_rows, "L9 Only", full_weights),
            "Proposed": method_summary(proposed_rows, "Proposed", full_weights),
            "Random Search": {
                "method": "Random Search",
                "trials": random_trials,
                "best_score_mean": round(sum(random_best_scores) / len(random_best_scores), 6) if random_best_scores else None,
                "best_score_std": _std(random_best_scores),
                "best_examples": random_bests[:5],
            },
            "proposed_meta": proposed_meta,
        }
    return report


def _std(values):
    if not values:
        return None
    mean = sum(values) / len(values)
    return round(math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)), 6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/results.json")
    parser.add_argument("--output", default="results/offline_comparison.json")
    parser.add_argument("--budgets", default=",".join(str(x) for x in BUDGETS))
    parser.add_argument("--random_trials", type=int, default=30)
    args = parser.parse_args()

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    _, by_config = load_results(args.results)
    report = compare_methods(by_config, budgets, random_trials=args.random_trials)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("[offline] saved {0}".format(args.output))


if __name__ == "__main__":
    main()
