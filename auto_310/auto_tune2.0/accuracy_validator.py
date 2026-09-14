# accuracy_validator.py
# 对比ONNX模型（本地）与OM模型（昇腾310B）的输出余弦相似度
# 依赖：pip install onnxruntime numpy

import numpy as np
import onnxruntime as ort
import json
import os
import tempfile
from npu_connector import NPUConnector


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    a_flat = a.flatten().astype(np.float32)
    b_flat = b.flatten().astype(np.float32)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def run_onnx_infer(onnx_path: str, input_data: np.ndarray) -> np.ndarray:
    """在本地用onnxruntime运行ONNX推理"""
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    return outputs[0]  # 取第一个输出


def run_om_infer_via_msame(
    connector: NPUConnector,
    om_path: str,
    input_data: np.ndarray,
    remote_workdir: str = "/home/HwHiAiUser/msame_test"
) -> np.ndarray:
    """
    通过SSH在310B上用msame运行OM推理，结果回传本地。
    msame需要预先安装在310B上：https://gitee.com/ascend/tools/tree/master/msame
    """
    # 将输入数据保存为bin文件并上传（每个样本只上传输入，om模型由外层复用）
    with tempfile.TemporaryDirectory() as tmpdir:
        input_bin = os.path.join(tmpdir, "input_0.bin")
        input_data.astype(np.float32).tofile(input_bin)
        # 清空上次输出，避免find到旧结果
        connector.exec(f"rm -rf {remote_workdir}/output && mkdir -p {remote_workdir}/output")
        connector.upload(input_bin, f"{remote_workdir}/input/input_0.bin")

    # om模型路径直接用remote_workdir拼接，不再重复上传
    om_basename = os.path.basename(om_path)
    remote_om = f"{remote_workdir}/{om_basename}.om"
    # 清空上次推理的输出，避免find找到旧文件
    connector.exec(f"rm -rf {remote_workdir}/output && mkdir -p {remote_workdir}/output")

    MSAME_PATH = "/home/HwHiAiUser/AscendProjects/tools/msame/out/msame"
    ACL_LIB    = "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64"
    # 执行msame推理（显式设置LD_LIBRARY_PATH，SSH非交互会话不加载.bashrc）
    cmd = (
        f"export LD_LIBRARY_PATH={ACL_LIB}:$LD_LIBRARY_PATH && "
        f"{MSAME_PATH} --model {remote_om} "
        f"--input {remote_workdir}/input/input_0.bin "
        f"--output {remote_workdir}/output "
        f"--outfmt BIN 2>&1"
    )
    print(f"[NPU] 执行推理命令: {cmd}")
    stdout, stderr, rc = connector.exec(cmd, timeout=60)
    if rc != 0:
        # 命令末尾使用了2>&1，msame的错误信息通常已经进入stdout。
        detail = stdout.strip() or stderr.strip() or "<无错误输出>"
        raise RuntimeError(
            f"msame推理失败，returncode={rc}\n{detail}"
        )

    # 下载输出结果
    # msame会在output下创建带时间戳的子目录，用find递归查找第一个bin文件
    with tempfile.TemporaryDirectory() as tmpdir:
        local_out = os.path.join(tmpdir, "output_0.bin")
        find_out, _, _ = connector.exec(
            f"find {remote_workdir}/output -name '*.bin' | sort | tail -1"
        )
        remote_out = find_out.strip()
        if not remote_out:
            raise RuntimeError("msame未生成输出文件，请检查推理是否成功")
        connector.download(remote_out, local_out)
        result = np.fromfile(local_out, dtype=np.float32)

    return result


def validate_accuracy(
    onnx_path: str,
    om_path: str,
    input_shape: tuple,
    connector: NPUConnector,
    num_samples: int = 10,
    seed: int = 42
) -> dict:
    """
    主函数：对比ONNX与OM模型在多个随机输入下的余弦相似度。

    Args:
        onnx_path: 本地ONNX模型路径
        om_path: 本地OM模型路径（不含.om后缀，与ATC输出一致）
        input_shape: 输入shape，如 (1, 3, 224, 224)
        connector: NPUConnector实例
        num_samples: 测试样本数量
        seed: 随机种子

    Returns:
        {
          "cosine_mean": float,     # 平均余弦相似度
          "cosine_min": float,      # 最低余弦相似度（最差情况）
          "cosine_max": float,      # 最高余弦相似度
          "samples": [...]          # 每个样本的详情
        }
    """
    rng = np.random.default_rng(seed)
    sample_results = []

    print(f"[精度验证] 开始，共 {num_samples} 个测试样本")
    print(f"  ONNX: {onnx_path}")
    print(f"  OM:   {om_path}.om")

    # om模型只上传一次，10个样本复用
    om_basename = os.path.basename(om_path)
    remote_workdir = "/home/HwHiAiUser/msame_test"
    remote_om = f"{remote_workdir}/{om_basename}.om"
    connector.exec(f"mkdir -p {remote_workdir}/input {remote_workdir}/output")
    connector.upload(om_path + ".om", remote_om)
    print(f"  OM已上传至310B: {remote_om}")

    for i in range(num_samples):
        # 生成随机输入（实际使用中可替换为真实测试数据）
        input_data = rng.standard_normal(input_shape).astype(np.float32)

        onnx_output = run_onnx_infer(onnx_path, input_data)
        om_output = run_om_infer_via_msame(connector, om_path, input_data, remote_workdir=remote_workdir)

        cos_sim = cosine_similarity(onnx_output, om_output)
        sample_results.append({
            "sample_id": i,
            "cosine_similarity": round(cos_sim, 6)
        })
        print(f"  样本 {i+1:2d}: 余弦相似度 = {cos_sim:.6f}")

    cosine_values = [s["cosine_similarity"] for s in sample_results]
    result = {
        "cosine_mean": round(float(np.mean(cosine_values)), 6),
        "cosine_min":  round(float(np.min(cosine_values)), 6),
        "cosine_max":  round(float(np.max(cosine_values)), 6),
        "input_shape": list(input_shape),
        "actual_batch_size": int(input_shape[0]),
        "num_samples": num_samples,
        "random_seed": seed,
        "samples": sample_results
    }

    print(f"\n[精度验证] 结果汇总:")
    print(f"  平均余弦相似度: {result['cosine_mean']}")
    print(f"  最低余弦相似度: {result['cosine_min']}")
    return result
