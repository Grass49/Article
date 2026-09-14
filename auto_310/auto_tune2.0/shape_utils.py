"""Input-shape helpers shared by compilation, validation and inference."""

def parse_atc_input_shape(input_shape):
    """Parse ``name:N,C,H,W`` and validate that every dimension is static."""
    if ":" not in input_shape:
        raise ValueError(f"输入shape格式错误，应为 name:N,C,H,W，实际为: {input_shape}")
    name, dims_text = input_shape.split(":", 1)
    name = name.strip()
    if not name:
        raise ValueError("输入shape缺少输入节点名称")
    try:
        dims = tuple(int(value.strip()) for value in dims_text.split(","))
    except ValueError as exc:
        raise ValueError(f"输入shape包含非整数维度: {input_shape}") from exc
    if len(dims) < 2 or any(value <= 0 for value in dims):
        raise ValueError(f"输入shape维度必须为正整数且至少包含Batch和特征维: {input_shape}")
    return name, dims


def input_shape_for_batch(input_shape, batch_size):
    """Replace the first dimension with the actual experiment batch size."""
    if batch_size <= 0:
        raise ValueError(f"Batch Size必须为正整数，实际为: {batch_size}")
    name, dims = parse_atc_input_shape(input_shape)
    actual_dims = (batch_size, *dims[1:])
    return f"{name}:" + ",".join(str(value) for value in actual_dims)


def shape_tuple(input_shape):
    """Return only the numeric dimensions from an ATC input-shape string."""
    return parse_atc_input_shape(input_shape)[1]
