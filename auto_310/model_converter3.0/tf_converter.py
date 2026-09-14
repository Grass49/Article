#!/usr/bin/env python3
# tf_converter.py — 容器内执行脚本：TensorFlow SavedModel → ONNX
"""
用法（容器内）：
    python tf_converter.py \
        --input  /workspace/input/saved_model \
        --output /workspace/onnx/model.onnx \
        [--opset 11] [--no-fold-const]
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def convert(input_path: str, output_path: str, opset: int = 11, fold_const: bool = True) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", input_path,
        "--output",      output_path,
        "--opset",       str(opset),
    ]
    if fold_const:
        cmd.append("--fold_const")

    print(f"[TensorFlow] 调用 tf2onnx: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("tf2onnx 转换失败")
    print(f"[TensorFlow] ✓ 导出完成 → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",         required=True)
    parser.add_argument("--output",        required=True)
    parser.add_argument("--shape",         default="1,224,224,3")  # 保持接口统一，tf2onnx 自动推断
    parser.add_argument("--opset",         type=int, default=11)
    parser.add_argument("--no-fold-const", action="store_true")
    args = parser.parse_args()

    convert(args.input, args.output, args.opset, not args.no_fold_const)
