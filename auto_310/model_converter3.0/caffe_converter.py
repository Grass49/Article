#!/usr/bin/env python3
# caffe_converter.py — 容器内执行脚本：Caffe → ONNX
"""
用法（容器内）：
    python caffe_converter.py \
        --input   /workspace/input/vgg16.caffemodel \
        --output  /workspace/onnx/vgg16.onnx \
        [--prototxt /workspace/input/vgg16.prototxt]
"""
from __future__ import annotations
import argparse
from pathlib import Path


def _resolve_prototxt(caffemodel: str, prototxt: str | None) -> str:
    if prototxt:
        return prototxt
    # 同名 .prototxt
    guess = str(Path(caffemodel).with_suffix(".prototxt"))
    if Path(guess).exists():
        return guess
    # deploy.prototxt
    deploy = str(Path(caffemodel).parent / "deploy.prototxt")
    if Path(deploy).exists():
        return deploy
    raise FileNotFoundError("未找到 .prototxt，请通过 --prototxt 显式指定")


def convert(input_path: str, output_path: str, prototxt: str | None = None) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    proto = _resolve_prototxt(input_path, prototxt)

    print(f"[Caffe] prototxt  : {proto}")
    print(f"[Caffe] caffemodel: {input_path}")

    try:
        from caffe2onnx.src.load_save_model import loadcaffemodel, saveonnxmodel
        from caffe2onnx.src.caffe2onnx import Caffe2Onnx
        graph, params = loadcaffemodel(proto, input_path)
        onnx_model = Caffe2Onnx(graph, params, output_path).createOnnxModel()
        saveonnxmodel(onnx_model, output_path)
    except ImportError:
        import subprocess, sys
        ir = str(Path(output_path).with_suffix(""))
        subprocess.run([sys.executable, "-m", "mmdnn.conversion._script.convertToIR",
                        "-f", "caffe", "-n", proto, "-w", input_path, "-d", ir], check=True)
        subprocess.run([sys.executable, "-m", "mmdnn.conversion._script.IRToCode",
                        "-f", "onnx", "--IRModelPath", f"{ir}.pb",
                        "--IRWeightPath", f"{ir}.npy", "-o", output_path], check=True)

    print(f"[Caffe] ✓ 导出完成 → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    required=True)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--shape",    default="1,3,224,224")  # 保持接口统一
    parser.add_argument("--opset",    type=int, default=11)
    parser.add_argument("--prototxt", default=None)
    args = parser.parse_args()

    convert(args.input, args.output, args.prototxt)
