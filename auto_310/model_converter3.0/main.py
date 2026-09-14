#!/usr/bin/env python3
# main.py — 控制层：通过 docker exec 调用容器内转换脚本
"""
用法：
    python main.py
"""
from __future__ import annotations
import subprocess
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath

# ── 容器配置 ──────────────────────────────────────────────────────────────
# 宿主机共享目录（Windows 路径）→ 容器内挂载点
HOST_WORKSPACE = Path(r"E:\workspace")
CONTAINER_WORKSPACE = "/workspace"

# 框架 → 容器名
CONTAINERS: dict[str, str] = {
    "pytorch":     "torch_container",
    "tensorflow":  "tf_container",
    "paddle":      "paddle_container",
    "caffe":       "caffe_container",
}

# 框架 → 容器内转换脚本路径（脚本需已复制到容器镜像中）
SCRIPTS: dict[str, str] = {
    "pytorch":    "/app/pytorch_converter.py",
    "tensorflow": "/app/tf_converter.py",
    "paddle":     "/app/paddle_converter.py",
    "caffe":      "/app/caffe_converter.py",
}

DEFAULT_OPSET = 11


# ── 路径转换 ──────────────────────────────────────────────────────────────

def to_container_path(windows_path: str) -> str:
    """将宿主机 Windows 路径转换为容器内 POSIX 路径。
    例：E:\\workspace\\input\\resnet50.pth → /workspace/input/resnet50.pth
    """
    p = Path(windows_path).resolve()
    try:
        rel = p.relative_to(HOST_WORKSPACE)
    except ValueError:
        raise ValueError(
            f"路径 '{windows_path}' 不在共享目录 '{HOST_WORKSPACE}' 下，"
            "请将模型放入 E:\\workspace 后重试"
        )
    return str(PurePosixPath(CONTAINER_WORKSPACE) / rel.as_posix())


def onnx_container_path(input_container_path: str) -> str:
    """根据输入容器路径推导 ONNX 输出路径（/workspace/onnx/<stem>.onnx）"""
    stem = PurePosixPath(input_container_path).stem
    return f"{CONTAINER_WORKSPACE}/onnx/{stem}.onnx"


# ── Docker 调用封装 ───────────────────────────────────────────────────────

def run_in_container(container: str, cmd: list[str]) -> None:
    """在指定容器内执行命令，实时输出日志，失败则抛出异常。"""
    full_cmd = ["docker", "exec", container] + cmd
    print(f"\n  $ {' '.join(full_cmd)}\n")
    result = subprocess.run(full_cmd)
    if result.returncode != 0:
        raise RuntimeError(f"容器 '{container}' 执行失败（exit {result.returncode}）")


# ── 交互逻辑 ──────────────────────────────────────────────────────────────

def _prompt(prompt: str, default: str | None = None) -> str:
    hint = f" [{default}]" if default else ""
    raw = input(f"{prompt}{hint}: ").strip()
    return raw if raw else (default or "")


def _pick_framework() -> str:
    keys = list(CONTAINERS.keys())
    print("\n支持的框架：")
    for i, name in enumerate(keys, 1):
        print(f"  {i}. {name.capitalize()}")
    print()
    while True:
        raw = input("请选择框架编号（或名称）: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(keys):
            return keys[int(raw) - 1]
        if raw.lower() in CONTAINERS:
            return raw.lower()
        print(f"  ✗ 无效输入，请重试")


def main() -> None:
    print("=" * 60)
    print("   模型 → ONNX 转换工具（Docker 容器化版）")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 选择框架
    framework = _pick_framework()
    container = CONTAINERS[framework]
    script    = SCRIPTS[framework]

    # 2. 输入模型路径（Windows 路径）
    win_path = _prompt("模型路径（Windows 路径，如 E:\\workspace\\input\\model.pth）")
    if not win_path:
        print("✗ 路径不能为空"); sys.exit(1)

    try:
        input_container = to_container_path(win_path)
    except ValueError as e:
        print(f"✗ {e}"); sys.exit(1)

    # 3. 输出路径（自动推导，可覆盖）
    default_output = onnx_container_path(input_container)
    output_container = _prompt("输出 ONNX 容器路径（留空自动生成）", default=default_output) or default_output

    # 4. Input shape
    default_shape = "1,224,224,3" if framework == "tensorflow" else "1,3,224,224"
    shape = _prompt("Input shape", default=default_shape)

    # 5. Opset
    opset = _prompt("ONNX opset 版本", default=str(DEFAULT_OPSET))

    # 6. 构造容器命令
    cmd = [
        "python", script,
        "--input",  input_container,
        "--output", output_container,
        "--shape",  shape,
        "--opset",  opset,
    ]

    # 框架专属参数
    if framework == "pytorch":
        dyn = _prompt("开启动态 batch？(y/n)", default="y").lower()
        if dyn == "n":
            cmd.append("--no-dynamic-batch")
    elif framework == "tensorflow":
        fc = _prompt("启用 fold_const？(y/n)", default="y").lower()
        if fc == "n":
            cmd.append("--no-fold-const")
    elif framework == "caffe":
        proto_win = _prompt("prototxt Windows 路径（留空自动查找）", default=None)
        if proto_win:
            try:
                cmd += ["--prototxt", to_container_path(proto_win)]
            except ValueError as e:
                print(f"✗ {e}"); sys.exit(1)

    # 7. 执行
    print(f"\n  容器    : {container}")
    print(f"  输入    : {input_container}")
    print(f"  输出    : {output_container}")

    try:
        run_in_container(container, cmd)
        print(f"\n✅ 转换完成！ONNX 文件位于容器路径: {output_container}")
        # 对应宿主机路径
        # rel = output_container.removeprefix(CONTAINER_WORKSPACE + "/")
        
        prefix = CONTAINER_WORKSPACE + "/"
        if output_container.startswith(prefix):
            rel = output_container[len(prefix):]
        else:
            rel = output_container
        
        host_out = HOST_WORKSPACE / rel.replace("/", "\\")
        print(f"   宿主机路径: {host_out}\n")
    except RuntimeError as e:
        print(f"\n✗ {e}"); sys.exit(1)


if __name__ == "__main__":
    main()
