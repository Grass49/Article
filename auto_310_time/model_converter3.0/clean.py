import onnx

model = onnx.load("E:/workspace/onnx/resnet-50-tensorflow2-classification-v1.onnx")

print("IR version:", model.ir_version)

print("\nOpset imports:")
for i, opset in enumerate(model.opset_import):
    print(
        f"[{i}] domain={repr(opset.domain)}, "
        f"version={opset.version}"
    )

