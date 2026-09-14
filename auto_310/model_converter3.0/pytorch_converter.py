#!/usr/bin/env python3
# pytorch_converter.py — 容器内执行脚本：PyTorch → ONNX
"""
用法（容器内）：
    python pytorch_converter.py \
        --input  /workspace/input/resnet50.pth \
        --output /workspace/onnx/resnet50.onnx \
        --shape  1,3,224,224 \
        [--opset 11] [--no-dynamic-batch]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch


def convert(input_path: str, output_path: str, shape: list[int],
            opset: int = 11, dynamic_batch: bool = True) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[PyTorch] 加载模型: {input_path}")
    model = torch.load(input_path, map_location="cpu")
    model.eval()

    dummy = torch.randn(*shape)
    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}} if dynamic_batch else None

    torch.onnx.export(
        model, dummy, output_path,
        input_names=["input"], output_names=["output"],
        opset_version=opset, dynamic_axes=dynamic_axes,
    )
    print(f"[PyTorch] ✓ 导出完成 → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",            required=True)
    parser.add_argument("--output",           required=True)
    parser.add_argument("--shape",            default="1,3,224,224")
    parser.add_argument("--opset",            type=int, default=11)
    parser.add_argument("--no-dynamic-batch", action="store_true")
    args = parser.parse_args()

    shape = [int(x) for x in args.shape.split(",")]
    convert(args.input, args.output, shape, args.opset, not args.no_dynamic_batch)