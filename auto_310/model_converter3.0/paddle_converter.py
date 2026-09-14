#!/usr/bin/env python3
# paddle_converter.py — 容器内执行脚本：PaddlePaddle → ONNX
"""
用法（容器内）：
    python paddle_converter.py \
        --input  /workspace/input/paddle_model \
        --output /workspace/onnx/model.onnx \
        [--opset 11]
"""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path


def convert(input_path: str, output_path: str, opset: int = 11,
            model_filename: str = "model.pdmodel",
            params_filename: str = "model.pdiparams") -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[Paddle] 加载模型目录: {input_path}")
    cmd = [
        "paddle2onnx",
        "--model_dir",        input_path,
        "--model_filename",   model_filename,
        "--params_filename",  params_filename,
        "--save_file",        output_path,
        "--opset_version",    str(opset),
        "--enable_onnx_checker", "True",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("paddle2onnx 转换失败")
    print(f"[Paddle] ✓ 导出完成 → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",           required=True)
    parser.add_argument("--output",          required=True)
    parser.add_argument("--shape",           default="1,3,224,224")  # 保持接口统一
    parser.add_argument("--opset",           type=int, default=11)
    parser.add_argument("--model-filename",  default="model.pdmodel")
    parser.add_argument("--params-filename", default="model.pdiparams")
    args = parser.parse_args()

    convert(args.input, args.output, args.opset, args.model_filename, args.params_filename)
