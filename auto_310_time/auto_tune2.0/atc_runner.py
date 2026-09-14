# atc_runner.py
import os
import subprocess

from param_space import (
    BATCH_SIZE,
    GRAPH_OPTIMIZATION_STRATEGY,
    INPUT_FORMAT_FIXED,
    OP_SELECT_IMPLMODE,
    PRECISION_MODE,
    decode_config,
)
from shape_utils import input_shape_for_batch, shape_tuple


def build_atc_command(exp_id, cfg, onnx_path, base_input_shape):
    """Build a static-batch ATC command for one experiment configuration."""
    batch_size = BATCH_SIZE[cfg["B"]]
    graph_strategy = GRAPH_OPTIMIZATION_STRATEGY[cfg["D"]]
    actual_input_shape = input_shape_for_batch(base_input_shape, batch_size)
    om_path = "results/om_models/model_exp{0}".format(exp_id)

    cmd = [
        "atc",
        "--model={0}".format(onnx_path),
        '--input_shape="{0}"'.format(actual_input_shape),
        "--framework=5",
        "--soc_version=Ascend310B1",
        "--output={0}".format(om_path),
        "--precision_mode={0}".format(PRECISION_MODE[cfg["A"]]),
        "--op_select_implmode={0}".format(OP_SELECT_IMPLMODE[cfg["C"]]),
        "--buffer_optimize={0}".format(graph_strategy["buffer_optimize"]),
        "--tiling_schedule_optimize={0}".format(
            graph_strategy["tiling_schedule_optimize"]
        ),
        "--enable_single_stream={0}".format(
            str(graph_strategy["enable_single_stream"]).lower()
        ),
        "--input_format={0}".format(INPUT_FORMAT_FIXED),
    ]
    return cmd, om_path, actual_input_shape, batch_size


def run_atc(exp_id, cfg, onnx_path, input_shape):
    os.makedirs("results/om_models", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)

    cmd, om_path, actual_input_shape, batch_size = build_atc_command(
        exp_id, cfg, onnx_path, input_shape
    )

    cmd_str = " ".join(cmd)
    print("[ATC] Exp {0}: {1}".format(exp_id, cmd_str))

    env = os.environ.copy()
    env["TE_PARALLEL_COMPILER"] = "2"

    result = subprocess.run(
        cmd_str,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    log_path = "results/logs/atc_exp{0}.log".format(exp_id)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(result.stdout)
        f.write("\n")
        f.write(result.stderr)

    common = {
        "command": cmd_str,
        "actual_input_shape": actual_input_shape,
        "input_shape_tuple": shape_tuple(actual_input_shape),
        "batch_size": batch_size,
        "decoded_config": decode_config(cfg),
        "atc_log": log_path,
    }
    if result.returncode != 0:
        print("[ATC] Exp {0} failed, see {1}".format(exp_id, log_path))
        common.update({"success": False, "om_path": None, "error": result.stderr})
        return common

    common.update({"success": True, "om_path": om_path})
    return common
