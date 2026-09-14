import unittest

from atc_runner import build_atc_command
from evaluator import compute_scores
from metric_collector import collect_metrics
from orthogonal_table import ORTHOGONAL_L9, validate_l9
from shape_utils import input_shape_for_batch, shape_tuple


class ExperimentDesignTests(unittest.TestCase):
    def test_l9_is_balanced_and_pairwise_orthogonal(self):
        self.assertTrue(validate_l9())
        for factor in "ABCD":
            self.assertEqual(
                [row[factor] for row in ORTHOGONAL_L9].count(1), 3
            )
            self.assertEqual(
                [row[factor] for row in ORTHOGONAL_L9].count(2), 3
            )
            self.assertEqual(
                [row[factor] for row in ORTHOGONAL_L9].count(3), 3
            )

    def test_batch_replaces_actual_atc_shape(self):
        self.assertEqual(
            input_shape_for_batch("input:1,3,224,224", 16),
            "input:16,3,224,224",
        )
        self.assertEqual(shape_tuple("input:4,3,224,224"), (4, 3, 224, 224))

    def test_all_batch_levels_enter_atc_command(self):
        expected = {1: 1, 2: 4, 3: 16}
        for level, batch in expected.items():
            cfg = {"A": 1, "B": level, "C": 1, "D": 1}
            cmd, _, actual_shape, actual_batch = build_atc_command(
                level, cfg, "model.onnx", "input:1,3,224,224"
            )
            self.assertEqual(actual_batch, batch)
            self.assertEqual(actual_shape, f"input:{batch},3,224,224")
            self.assertIn(
                f'--input_shape="input:{batch},3,224,224"', cmd
            )

    def test_graph_optimization_levels_enter_atc_command(self):
        expected = {
            1: ("l2_optimize", "0", "false"),
            2: ("l2_optimize", "1", "false"),
            3: ("l2_optimize", "0", "true"),
        }
        for level, values in expected.items():
            cfg = {"A": 1, "B": 1, "C": 1, "D": level}
            cmd, _, _, _ = build_atc_command(
                level, cfg, "model.onnx", "input:1,3,224,224"
            )
            self.assertIn("--buffer_optimize={0}".format(values[0]), cmd)
            self.assertIn(
                "--tiling_schedule_optimize={0}".format(values[1]), cmd
            )
            self.assertIn(
                "--enable_single_stream={0}".format(values[2]), cmd
            )
            self.assertFalse(
                any(item.startswith("--topo_sorting_mode=") for item in cmd)
            )


class MetricAndEvaluatorTests(unittest.TestCase):
    def test_metric_collector_does_not_invent_accuracy(self):
        metrics = collect_metrics(8.0, 500.0, 4)
        self.assertNotIn("accuracy_drop", metrics)
        self.assertEqual(metrics["actual_batch_size"], 4)
        self.assertEqual(metrics["equivalent_sample_latency_ms"], 2.0)

    def test_missing_resource_metrics_are_excluded(self):
        rows = [
            {
                "exp_id": 1,
                "config": {},
                "metrics": {"fps": 100.0, "latency_ms": 10.0, "cosine_mean": 1.0},
            },
            {
                "exp_id": 2,
                "config": {},
                "metrics": {"fps": 200.0, "latency_ms": 10.0, "cosine_mean": 1.0},
            },
        ]
        scored, weights = compute_scores(rows)
        self.assertEqual(weights, {"fps": 1.0})
        self.assertEqual(scored[0]["score"], 0.0)
        self.assertEqual(scored[1]["score"], 1.0)

    def test_cosine_is_constraint_not_minmax_objective(self):
        rows = [
            {
                "exp_id": 1,
                "config": {},
                "metrics": {"fps": 100.0, "latency_ms": 10.0, "cosine_mean": 0.998},
            },
            {
                "exp_id": 2,
                "config": {},
                "metrics": {"fps": 90.0, "latency_ms": 11.0, "cosine_mean": 0.999},
            },
        ]
        scored, _ = compute_scores(rows)
        self.assertFalse(scored[0]["feasible"])
        self.assertEqual(scored[0]["score"], -1.0)
        self.assertTrue(scored[1]["feasible"])

    def test_available_objectives_receive_equal_weights(self):
        rows = [
            {
                "exp_id": 1,
                "config": {},
                "metrics": {
                    "fps": 100.0,
                    "latency_ms": 10.0,
                    "cosine_mean": 1.0,
                    "perf_stats": {"die_temp_avg": 50.0, "memory_used_mb_avg": 100.0},
                },
            },
            {
                "exp_id": 2,
                "config": {},
                "metrics": {
                    "fps": 200.0,
                    "latency_ms": 10.0,
                    "cosine_mean": 1.0,
                    "perf_stats": {"die_temp_avg": 60.0, "memory_used_mb_avg": 200.0},
                },
            },
        ]
        _, weights = compute_scores(rows)
        self.assertEqual(
            weights,
            {"fps": 0.333333, "temperature": 0.333333, "memory_mb": 0.333333},
        )


if __name__ == "__main__":
    unittest.main()
