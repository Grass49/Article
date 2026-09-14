# param_space.py

# Factor A: precision mode
PRECISION_MODE = {
    1: "force_fp32",
    2: "allow_fp32_to_fp16",
    3: "force_fp16",
}

# Factor B: batch size
BATCH_SIZE = {
    1: 1,
    2: 4,
    3: 16,
}

# Factor C: operator implementation mode
OP_SELECT_IMPLMODE = {
    1: "high_precision",
    2: "high_performance",
    3: "high_performance_for_all",
}

# Factor D: graph optimization strategy.  Each level maps to a distinct
# combination supported by the CANN 8.0.RC2 ATC used in this project.
GRAPH_OPTIMIZATION_STRATEGY = {
    1: {
        "name": "standard_l2",
        "buffer_optimize": "l2_optimize",
        "tiling_schedule_optimize": 0,
        "enable_single_stream": False,
    },
    2: {
        "name": "l2_with_tiling",
        "buffer_optimize": "l2_optimize",
        "tiling_schedule_optimize": 1,
        "enable_single_stream": False,
    },
    3: {
        "name": "l2_single_stream",
        "buffer_optimize": "l2_optimize",
        "tiling_schedule_optimize": 0,
        "enable_single_stream": True,
    },
}

INPUT_FORMAT_FIXED = "NCHW"
FACTOR_LEVELS = ("A", "B", "C", "D")
LEVEL_VALUES = (1, 2, 3)

FACTOR_NAMES = {
    "A": "precision_mode",
    "B": "batch_size",
    "C": "op_select_implmode",
    "D": "graph_optimization_strategy",
}


def decode_config(cfg):
    strategy = GRAPH_OPTIMIZATION_STRATEGY[cfg["D"]]
    return {
        "precision_mode": PRECISION_MODE[cfg["A"]],
        "batch_size": BATCH_SIZE[cfg["B"]],
        "op_select_implmode": OP_SELECT_IMPLMODE[cfg["C"]],
        "graph_optimization_strategy": strategy["name"],
        "buffer_optimize": strategy["buffer_optimize"],
        "tiling_schedule_optimize": strategy["tiling_schedule_optimize"],
        "enable_single_stream": strategy["enable_single_stream"],
    }


def full_factorial_space():
    rows = []
    for a in LEVEL_VALUES:
        for b in LEVEL_VALUES:
            for c in LEVEL_VALUES:
                for d in LEVEL_VALUES:
                    rows.append({"A": a, "B": b, "C": c, "D": d})
    return rows


FULL_FACTORIAL_81 = full_factorial_space()
