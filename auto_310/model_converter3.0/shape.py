import onnx

m = onnx.load("E:/workspace/onnx/resnet18_full.onnx")

for x in m.graph.input:
    print("Input name:", x.name)

    dims = []
    shape = x.type.tensor_type.shape

    for dim in shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")

    print("Shape:", dims)