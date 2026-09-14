import onnx

model_path = r"E:/workspace/onnx/PPLCNet_x1_0_infer_paddle.onnx"

model = onnx.load(model_path)

print("========== Inputs ==========")

for tensor in model.graph.input:
    dims = []

    for dim in tensor.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")

    print(f"{tensor.name}: {dims}")


print("\n========== Outputs ==========")

for tensor in model.graph.output:
    dims = []

    for dim in tensor.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")

    print(f"{tensor.name}: {dims}")