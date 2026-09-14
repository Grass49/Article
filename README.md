# Budget-Constrained Multi-Stage Search for Deep Learning Deployment Configurations on the Ascend 310B Edge NPU

This repository contains the code, validation utilities, experiment artifacts, and result data used to study budget-constrained multi-stage search for deep-learning deployment configurations on the Huawei Ascend 310B edge NPU.

## Contents

- `auto_310/`: model conversion and automatic configuration-search pipeline
- `auto_310_time/`: timing-instrumented version of the search pipeline
- `comment6_validation/`: ONNX/OM validation scripts and environment requirements
- `om_onnx_exp/`: ONNX-to-OM experimental materials and evaluation data
- `paddle/`: Paddle-related experiment files
- `logs/`: experiment logs
- `*.csv`: raw experimental measurements and derived scores

## Usage

Start with the scripts and documentation in the relevant subdirectory. Ascend-side experiments require a compatible Ascend 310B environment and CANN toolchain; validation dependencies are listed in `comment6_validation/requirements_pc.txt` and `comment6_validation/requirements_npu.txt`.

## Citation

If you use this repository, please cite the accompanying paper, *Budget-Constrained Multi-Stage Search for Deep Learning Deployment Configurations on the Ascend 310B Edge NPU*.

## License

License information will be added before release.
