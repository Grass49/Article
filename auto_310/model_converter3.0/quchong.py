import onnx

input_file = "E:/workspace/onnx/resnet-50-tensorflow2-classification-v1.onnx"
output_file = "E:/workspace/onnx/resnet-50-tensorflow2-classification-v1-quchong.onnx"

model = onnx.load(input_file)

# 只保留默认 ONNX domain
default_opset = None

for opset in model.opset_import:
    if opset.domain == "":
        default_opset = opset.version
        break

if default_opset is None:
    raise RuntimeError("找不到默认 ONNX domain")

del model.opset_import[:]

opset = model.opset_import.add()
opset.domain = ""
opset.version = default_opset

onnx.checker.check_model(model)
onnx.save(model, output_file)

print("Saved:", output_file)

for x in model.opset_import:
    print(
        f"domain={repr(x.domain)}, version={x.version}"
    )