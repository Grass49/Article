# orthogonal_table.py

ORTHOGONAL_L9 = [
    {"A":1, "B":1, "C":1, "D":1},
    {"A":1, "B":2, "C":2, "D":2},
    {"A":1, "B":3, "C":3, "D":3},
    {"A":2, "B":1, "C":2, "D":3},
    {"A":2, "B":2, "C":3, "D":1},
    {"A":2, "B":3, "C":1, "D":2},
    {"A":3, "B":1, "C":3, "D":2},
    {"A":3, "B":2, "C":1, "D":3},
    {"A":3, "B":3, "C":2, "D":1},
]


def validate_l9(table=ORTHOGONAL_L9):
    """Fail fast if a future edit breaks the L9(3^4) balance."""
    factors = ("A", "B", "C", "D")
    if len(table) != 9:
        raise ValueError(f"L9正交表必须包含9组实验，当前为{len(table)}组")
    for factor in factors:
        counts = {level: 0 for level in (1, 2, 3)}
        for row in table:
            counts[row[factor]] = counts.get(row[factor], 0) + 1
        if counts != {1: 3, 2: 3, 3: 3}:
            raise ValueError(f"因素{factor}水平不平衡: {counts}")
    for index, left in enumerate(factors):
        for right in factors[index + 1:]:
            pairs = {(row[left], row[right]) for row in table}
            if len(pairs) != 9:
                raise ValueError(f"因素{left}/{right}的水平组合不正交")
    return True


validate_l9()
