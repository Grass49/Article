#!/usr/bin/env python3
# gui_app.py — 自动化部署软件 GUI 主程序（Windows 本机运行）
# 运行：python gui_app.py

from __future__ import annotations
import subprocess
import sys
import os
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path, PureWindowsPath, PurePosixPath

ROOT = Path(__file__).parent

# ── 容器配置 ──────────────────────────────────────────────────────────────────
HOST_WORKSPACE   = Path(r"E:\workspace")   # Windows 宿主机共享目录
CONT_WORKSPACE   = "/workspace"            # 容器内挂载点

CONVERTER_CONTAINERS = {
    "PyTorch":     ("torch_container",  "/app/pytorch_converter.py"),
    "TensorFlow":  ("tf_container",     "/app/tf_converter.py"),
    "Caffe":       ("caffe_container",  "/app/caffe_converter.py"),
    "PaddlePaddle":("paddle_container", "/app/paddle_converter.py"),
}

ASCEND_CONTAINER      = "ascend"
ASCEND_AUTOTUNE_DIR   = "/home/AscendWork/auto_tune2.0"
ASCEND_PIPELINE_SCRIPT = f"{ASCEND_AUTOTUNE_DIR}/run_pipeline.py"

# ─────────────────────────────────────────────────────────────────────────────
# 日志队列
# ─────────────────────────────────────────────────────────────────────────────
log_queue: queue.Queue[str] = queue.Queue()

def log(msg: str):
    log_queue.put(str(msg))


def redact_sensitive_args(cmd: list[str]) -> list[str]:
    """Return a display-only command with credential values hidden."""
    redacted = list(cmd)
    for index, value in enumerate(redacted[:-1]):
        if value in {"--npu_pass", "--password", "-p"}:
            redacted[index + 1] = "******"
    return redacted


# ─────────────────────────────────────────────────────────────────────────────
# Docker 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def docker_exec_stream(container: str, cmd: list[str], stop_event: threading.Event):
    """
    在容器内执行命令，实时将 stdout/stderr 写入 log_queue。
    返回 returncode。
    """
    full = ["docker", "exec", "-i", "-u", "root", container] + cmd
    display_cmd = redact_sensitive_args(cmd)
    log(f"$ docker exec {container} {' '.join(display_cmd)}")
    proc = subprocess.Popen(
        full,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    def _read():
        for raw in proc.stdout:
            if stop_event.is_set():
                break
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                log(line)

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join()
    proc.wait()
    return proc.returncode


def docker_exec_quiet(container: str, cmd: list[str]) -> tuple[str, str, int]:
    """静默执行，返回 (stdout, stderr, rc)。"""
    r = subprocess.run(
        ["docker", "exec", container] + cmd,
        capture_output=True,
    )
    return (
        r.stdout.decode("utf-8", errors="replace"),
        r.stderr.decode("utf-8", errors="replace"),
        r.returncode,
    )


def docker_cp_to(local_path: str, container: str, remote_path: str) -> int:
    r = subprocess.run(
        ["docker", "cp", local_path, f"{container}:{remote_path}"],
        capture_output=True,
    )
    return r.returncode


def win_to_container(win_path: str) -> str:
    """E:\\workspace\\input\\model.pth  →  /workspace/input/model.pth"""
    p = Path(win_path).resolve()
    rel = p.relative_to(HOST_WORKSPACE)
    return str(PurePosixPath(CONT_WORKSPACE) / rel.as_posix())


# ─────────────────────────────────────────────────────────────────────────────
# 转换流程（后台线程）
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(cfg: dict, progress_cb, done_cb, stop_event: threading.Event):
    try:
        framework   = cfg["framework"]    # PyTorch / TensorFlow / Caffe / PaddlePaddle / ONNX
        model_path  = cfg["model_path"]   # Windows 本地路径
        input_shape = cfg["input_shape"]  # 如 "448,448,3"
        npu_host    = cfg["npu_host"]
        npu_port    = cfg["npu_port"]
        npu_user    = cfg["npu_user"]
        npu_pass    = cfg["npu_pass"]
        auto_tune   = cfg["auto_tune"]

        # ── 步骤 1：模型 → ONNX ──────────────────────────────────────────────
        if framework == "ONNX":
            # 已是 ONNX，直接用
            onnx_win = model_path
            log(f"跳过ONNX转换（已是ONNX格式）: {Path(model_path).name}")
        else:
            log(f"开始转换模型: {Path(model_path).name} ({framework}, Ascend310B4)...")
            onnx_win = _step_convert_to_onnx(framework, model_path, input_shape, stop_event)
            if onnx_win is None:
                done_cb(False); return
            log("ONNX转换成功")

        if stop_event.is_set():
            done_cb(False); return
        progress_cb(20)

        # ── 步骤 2：把 ONNX 复制进 ascend 容器 ───────────────────────────────
        onnx_name   = Path(onnx_win).name
        remote_onnx = f"{ASCEND_AUTOTUNE_DIR}/{onnx_name}"
        log(f"上传ONNX到ascend容器: {onnx_name}")
        rc = docker_cp_to(onnx_win, ASCEND_CONTAINER, remote_onnx)
        if rc != 0:
            log(f"[错误] docker cp 失败 (rc={rc})")
            done_cb(False); return

        # ── 步骤 3：在 ascend 容器内运行 run_pipeline.py ─────────────────────
        atc_shape = _shape_to_atc(input_shape)
        log("开始ONNX转OM模型...")

        cmd = [
            "python3", "-u", ASCEND_PIPELINE_SCRIPT,
            "--onnx",     remote_onnx,
            "--shape",    atc_shape,
            "--npu_host", npu_host,
            "--npu_port", str(npu_port),
            "--npu_user", npu_user,
            "--npu_pass", npu_pass,
        ]
        if not auto_tune:
            cmd.append("--no_tune")

        rc = docker_exec_stream(ASCEND_CONTAINER, cmd, stop_event)
        if rc != 0 and not stop_event.is_set():
            log(f"[错误] run_pipeline.py 退出码 {rc}")
            done_cb(False); return

        progress_cb(100)
        done_cb(True)

    except Exception as exc:
        import traceback
        log(f"[错误] {exc}")
        log(traceback.format_exc())
        done_cb(False)


def _step_convert_to_onnx(
    framework: str, model_path: str, input_shape: str,
    stop_event: threading.Event
) -> str | None:
    """
    在对应框架容器内执行转换，返回宿主机 ONNX 路径（在 E:\\workspace\\onnx\\）。
    失败返回 None。
    """
    container, script = CONVERTER_CONTAINERS[framework]

    # 模型路径必须在 E:\workspace 下
    try:
        input_cont = win_to_container(model_path)
    except ValueError as e:
        log(f"[错误] {e}")
        return None

    stem        = Path(model_path).stem
    output_cont = f"{CONT_WORKSPACE}/onnx/{stem}.onnx"

    # 构造 shape 参数
    parts = [int(x.strip()) for x in input_shape.split(",")]
    if len(parts) == 3:
        h, w, c = parts
        shape_str = f"1,{c},{h},{w}"   # NHWC → NCHW
    else:
        shape_str = ",".join(str(x) for x in parts)

    cmd = [
        "python", "-u", script,
        "--input",  input_cont,
        "--output", output_cont,
        "--shape",  shape_str,
        "--opset",  "11",
    ]

    rc = docker_exec_stream(container, cmd, stop_event)
    if rc != 0:
        log(f"[错误] {framework} 转换失败 (rc={rc})")
        return None

    # 对应宿主机路径
    # 兼容宿主机上的 Python 3.8：str.removeprefix() 从 Python 3.9 才提供。
    workspace_prefix = CONT_WORKSPACE.rstrip("/") + "/"
    rel = output_cont[len(workspace_prefix):] if output_cont.startswith(workspace_prefix) else output_cont
    return str(HOST_WORKSPACE / rel.replace("/", "\\"))


def _shape_to_atc(shape_str: str) -> str:
    """448,448,3  →  input:1,3,448,448"""
    parts = [int(x.strip()) for x in shape_str.split(",")]
    if len(parts) == 3:
        h, w, c = parts
        parts = [1, c, h, w]
    return "input:" + ",".join(str(x) for x in parts)


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

# 关键词 → 进度值（用于根据日志自动推进进度条）
_PROGRESS_KEYWORDS: list[tuple[str, int]] = [
    ("ONNX转换成功",       20),
    ("上传ONNX到ascend",   25),
    ("开始ONNX转OM模型",   30),
    ("OM模型转换成功",      40),
    ("开始模型精度评估",    50),
    ("模型精度评估完成",    60),
    ("开始模型性能评估",    70),
    ("模型性能评估完成",    80),
    ("综合最优模型",        95),
    ("模型转换与优化成功",  100),
]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("自动化部署软件")
        self.resizable(False, False)
        self._running    = False
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._build_ui()
        self._poll_log()

    # ── UI 构建 ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.configure(bg="#f0f0f0")

        left = tk.Frame(self, bg="#f0f0f0", padx=10, pady=10)
        left.grid(row=0, column=0, sticky="nsew")

        # 服务器设置
        self._section(left, "服务器设置", 0)
        tk.Label(left, text="开发套件 IP：", bg="#f0f0f0").grid(row=1, column=0, sticky="w")
        tk.Label(left, text="端口：",        bg="#f0f0f0").grid(row=1, column=1, sticky="w")
        self.var_ip   = tk.StringVar(value="192.168.137.100")
        self.var_port = tk.StringVar(value="22")
        tk.Entry(left, textvariable=self.var_ip,   width=16).grid(row=2, column=0, sticky="w", padx=(0,4))
        tk.Entry(left, textvariable=self.var_port, width=8 ).grid(row=2, column=1, sticky="w")
        tk.Button(left, text="检查连接", command=self._check_conn,
                  width=10).grid(row=2, column=2, padx=(6,0))

        # 模型信息
        self._section(left, "模型信息", 3)
        tk.Label(left, text="模型名称：", bg="#f0f0f0").grid(row=4, column=0, sticky="w", columnspan=3)
        self.var_name = tk.StringVar(value="UNet")
        tk.Entry(left, textvariable=self.var_name, width=36).grid(
            row=5, column=0, columnspan=3, sticky="ew")

        tk.Label(left, text="推理框架", bg="#f0f0f0").grid(
            row=6, column=0, sticky="w", columnspan=3, pady=(6,2))
        self.var_fw = tk.StringVar(value="PyTorch")
        for i, fw in enumerate(["ONNX", "Caffe", "TensorFlow", "PyTorch"]):
            tk.Radiobutton(left, text=fw, variable=self.var_fw, value=fw,
                           bg="#f0f0f0").grid(row=7+i, column=0, sticky="w", columnspan=3)

        tk.Label(left, text="输入尺寸（高度,宽度,通道）：", bg="#f0f0f0").grid(
            row=11, column=0, sticky="w", columnspan=3, pady=(6,0))
        self.var_shape = tk.StringVar(value="224,224,3")
        tk.Entry(left, textvariable=self.var_shape, width=36).grid(
            row=12, column=0, columnspan=3, sticky="ew")

        tk.Label(left, text="板卡类型", bg="#f0f0f0").grid(
            row=13, column=0, sticky="w", columnspan=3, pady=(6,0))
        self.var_chip = ttk.Combobox(left, values=["Ascend310B4"], state="readonly", width=33)
        self.var_chip.set("Ascend310B4")
        self.var_chip.grid(row=14, column=0, columnspan=3, sticky="ew")

        tk.Label(left, text="优化选项", bg="#f0f0f0").grid(
            row=15, column=0, sticky="w", columnspan=3, pady=(6,0))
        self.var_tune = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text="启用模型优化", variable=self.var_tune,
                       bg="#f0f0f0").grid(row=16, column=0, sticky="w", columnspan=3)

        # SSH 凭据（连接 310B 板卡）
        tk.Label(left, text="SSH 用户名：", bg="#f0f0f0").grid(
            row=17, column=0, sticky="w", pady=(6,0))
        tk.Label(left, text="SSH 密码：",   bg="#f0f0f0").grid(
            row=17, column=1, sticky="w", pady=(6,0), columnspan=2)
        self.var_user = tk.StringVar(value="HwHiAiUser")
        self.var_pass = tk.StringVar(value="Mind@123")
        tk.Entry(left, textvariable=self.var_user, width=14).grid(row=18, column=0, sticky="w")
        tk.Entry(left, textvariable=self.var_pass, show="*", width=18).grid(
            row=18, column=1, sticky="w", columnspan=2)

        # 文件传输
        self._section(left, "文件传输", 19)
        tk.Label(left, text="选择模型文件/目录：", bg="#f0f0f0").grid(
            row=20, column=0, sticky="w", columnspan=3)
        self.var_path = tk.StringVar(value="")
        tk.Entry(left, textvariable=self.var_path, width=36).grid(
            row=21, column=0, columnspan=3, sticky="ew")
        tk.Button(left, text="选择模型文件/目录", command=self._browse,
                  width=34).grid(row=22, column=0, columnspan=3, sticky="ew", pady=(4,0))

        # 进度条
        self.progress = ttk.Progressbar(left, length=260, mode="determinate", maximum=100)
        self.progress.grid(row=23, column=0, columnspan=2, sticky="ew", pady=(8,0))
        self.lbl_pct = tk.Label(left, text="0%", bg="#f0f0f0", width=5)
        self.lbl_pct.grid(row=23, column=2, sticky="w")

        # 按钮
        self.btn_start = tk.Button(left, text="开始转换", command=self._start,
                                   bg="#4CAF50", fg="white", width=34, relief="flat")
        self.btn_start.grid(row=24, column=0, columnspan=3, sticky="ew", pady=(8,2))

        self.btn_stop = tk.Button(left, text="停止转换", command=self._stop,
                                  bg="#cccccc", fg="#888888", width=34, relief="flat",
                                  state="disabled")
        self.btn_stop.grid(row=25, column=0, columnspan=3, sticky="ew", pady=(0,2))

        tk.Button(left, text="打开应用开发软件", command=self._open_dev,
                  width=34).grid(row=26, column=0, columnspan=3, sticky="ew")

        # 右侧日志面板
        right = tk.Frame(self, bg="#f0f0f0", padx=10, pady=10)
        right.grid(row=0, column=1, sticky="nsew")

        tk.Label(right, text="传输状态", bg="#f0f0f0",
                 font=("Microsoft YaHei", 10, "bold")).pack(anchor="w")

        frame_log = tk.Frame(right, bg="white", bd=1, relief="sunken")
        frame_log.pack(fill="both", expand=True)

        self.log_text = tk.Text(
            frame_log, width=62, height=30,
            bg="white", fg="#111111",
            font=("Consolas", 9),
            wrap="word", state="disabled", relief="flat",
        )
        sb = ttk.Scrollbar(frame_log, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)

        tk.Button(right, text="清空日志", command=self._clear_log,
                  width=60).pack(fill="x", pady=(6,0))

        self.columnconfigure(1, weight=1)

    def _section(self, parent, text, row):
        tk.Label(parent, text=text, bg="#f0f0f0",
                 font=("Microsoft YaHei", 9, "bold")).grid(
            row=row, column=0, sticky="w", columnspan=3, pady=(8,2))

    # ── 事件处理 ──────────────────────────────────────────────────────────────

    def _browse(self):
        path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[
                ("模型文件", "*.pth *.pt *.onnx *.pb *.caffemodel *.pdmodel"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            path = filedialog.askdirectory(title="或选择模型目录")
        if path:
            # 转换为 Windows 反斜杠路径
            self.var_path.set(str(Path(path)))

    def _check_conn(self):
        ip   = self.var_ip.get().strip()
        port = self.var_port.get().strip() or "22"
        user = self.var_user.get().strip() or "root"
        pw   = self.var_pass.get()

        def _do():
            # 通过 ascend 容器内的 python 测试 SSH 连通性
            cmd = [
                "python3", "-c",
                f"import paramiko; c=paramiko.SSHClient(); "
                f"c.set_missing_host_key_policy(paramiko.AutoAddPolicy()); "
                f"c.connect('{ip}', port={port}, username='{user}', password='{pw}'); "
                f"_,o,_=c.exec_command('echo ok'); print(o.read().decode()); c.close()"
            ]
            out, err, rc = docker_exec_quiet(ASCEND_CONTAINER, cmd)
            if rc == 0 and "ok" in out:
                log(f"[连接] {ip}:{port} 连接成功")
                self.after(0, lambda: messagebox.showinfo("连接成功", f"已成功连接到 {ip}:{port}"))
            else:
                msg = err.strip() or out.strip() or f"rc={rc}"
                log(f"[连接] 失败: {msg}")
                self.after(0, lambda: messagebox.showerror("连接失败", msg))

        threading.Thread(target=_do, daemon=True).start()

    def _start(self):
        if self._running:
            return
        model_path = self.var_path.get().strip()
        if not model_path:
            messagebox.showwarning("提示", "请先选择模型文件或目录")
            return

        cfg = {
            "framework":   self.var_fw.get(),
            "model_path":  model_path,
            "input_shape": self.var_shape.get().strip(),
            "npu_host":    self.var_ip.get().strip(),
            "npu_port":    self.var_port.get().strip() or "22",
            "npu_user":    self.var_user.get().strip() or "root",
            "npu_pass":    self.var_pass.get(),
            "auto_tune":   self.var_tune.get(),
        }

        self._running = True
        self._stop_event.clear()
        self._set_progress(0)
        self.btn_start.config(state="disabled", bg="#aaaaaa", fg="white")
        self.btn_stop.config(state="normal",   bg="#f44336", fg="white")

        self._worker = threading.Thread(
            target=run_pipeline,
            args=(cfg, self._set_progress, self._on_done, self._stop_event),
            daemon=True,
        )
        self._worker.start()

    def _stop(self):
        if self._running:
            self._stop_event.set()
            log("[用户] 已请求停止转换")

    def _on_done(self, success: bool):
        self._running = False
        self.after(0, lambda: self._reset_buttons(success))

    def _reset_buttons(self, success: bool):
        self.btn_start.config(state="normal", bg="#4CAF50", fg="white")
        self.btn_stop.config(state="disabled", bg="#cccccc", fg="#888888")
        if success:
            self._set_progress(100)

    def _set_progress(self, value: int):
        self.after(0, lambda v=value: self._update_progress(v))

    def _update_progress(self, value: int):
        self.progress["value"] = value
        self.lbl_pct.config(text=f"{value}%")

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    def _open_dev(self):
        os.startfile(str(ROOT))

    # ── 日志轮询 ──────────────────────────────────────────────────────────────

    def _poll_log(self):
        try:
            while True:
                msg = log_queue.get_nowait()
                self._append_log(msg)
                # 根据关键词自动推进进度条
                for kw, pct in _PROGRESS_KEYWORDS:
                    if kw in msg:
                        self._set_progress(pct)
                        break
        except queue.Empty:
            pass
        self.after(100, self._poll_log)

    def _append_log(self, msg: str):
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
