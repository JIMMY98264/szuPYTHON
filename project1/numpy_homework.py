"""
NumPy 数值计算练习 —— 参考答案（题目版 1）

对应题目：python/题目/project1_exercises.pdf
  NP-E1 ~ NP-E8  基础题
  NP-M1 ~ NP-M4  中等题

运行方式：
    python numpy_homework.py

约定：
  * 仅使用 Python 与 NumPy，开头 import numpy as np
  * 每题独立成函数，输入用题目给定数组
  * 全部结果由程序 realtime 计算，不写死数值
  * 展示保留 6 位小数（np.set_printoptions），核对允许绝对误差 1e-6
"""

import numpy as np

# 展示用：6 位小数、固定小数位、不用科学计数法（便于和题目要求对照核对）
np.set_printoptions(precision=6, floatmode="fixed", suppress=True)


def show(name, value):
    """统一打印：名字 = 值，并带上 shape / 类型信息。"""
    if isinstance(value, np.ndarray):
        print(f"{name} = {value}")
        print(f"    shape = {value.shape}, dtype = {value.dtype}")
    elif isinstance(value, (int, float, np.floating, np.integer)):
        print(f"{name} = {value:.6f}")
    else:
        print(f"{name} = {value}")


def title(text):
    print("\n" + "=" * 68)
    print(text)
    print("=" * 68)


# ======================================================================
# NP-E1 数组、形状与切片
# ======================================================================
def np_e1():
    A = np.array([[0.20, -0.50, 1.10, 0.40],
                  [0.80, 0.30, -0.70, 0.60],
                  [-0.40, 0.90, 0.10, -0.20]])

    print("A =")
    print(A)

    # 1) ndim / shape / size
    print(f"\n1) ndim = {A.ndim}   shape = {A.shape}   size = {A.size}")
    print("   ndim=2 说明是二维数组（3 行 4 列）；size = 3*4 = 12 个元素")

    # 2) 第 2 行（下标 1）与最后 2 列
    row1 = A[1]           # 整行 -> 一维，shape (4,)
    last2 = A[:, -2:]     # 所有行的最后 2 列 -> 二维，shape (3, 2)
    show("\n2) A[1]（第 2 行）", row1)
    show("   A[:, -2:]（最后 2 列）", last2)
    print("   注意：取单行会降维成一维 (4,)；取切片保持二维 (3, 2)")

    # 3) 复制后把负数改成 0，A 不变
    B = A.copy()          # 必须 copy，否则 B 与 A 共享内存，改 B 会改 A
    B[B < 0] = 0.0        # 布尔掩码赋值：只改负数的位置
    show("\n3) B = A.copy(); B[B < 0] = 0", B)
    print("   验证 A 未被修改（A 中仍有负数）：", bool((A < 0).any()))
    show("   A", A)


# ======================================================================
# NP-E2 逐元素乘法、点乘与长度
# ======================================================================
def np_e2():
    a = np.array([0.80, -0.60, 0.70])
    b = np.array([-0.20, 0.90, 0.40])

    # 1) a * b 是逐元素乘（还是个数组）；a @ b 是点乘（是一个数）
    ab = a * b
    dot = a @ b
    show("1) a * b（逐元素乘）", ab)
    show("   a @ b（点乘，标量）", dot)
    print(f"   a @ b 的 shape = {np.shape(dot)}  -> 空元组 ()，即 0 维标量")
    print("   a * b 每个分量 = a[i]*b[i]；a @ b 是全部乘积再求和")

    # 2) 平方 -> 求和 -> 开方
    len_a = np.sqrt(np.sum(a * a))
    len_b = np.sqrt(np.sum(b ** 2))
    show("\n2) ||a|| 用 sqrt(sum(a*a))", len_a)
    show("   ||b|| 用 sqrt(sum(b**2))", len_b)

    # 3) 用 np.linalg.norm 核对
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    show("3) np.linalg.norm(a)", norm_a)
    show("   np.linalg.norm(b)", norm_b)
    print("   两者一致：", np.allclose([len_a, len_b], [norm_a, norm_b]))


# ======================================================================
# NP-E3 两个向量的夹角
# ======================================================================
def angle_deg(a, b):
    """返回两个同维非零一维向量的夹角（度）。零向量 -> ValueError。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"维度不一致: {a.shape} vs {b.shape}")
    if a.ndim != 1:
        raise ValueError("只接受一维向量")
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("零向量没有确定方向，无法定义夹角")
    cos = (a @ b) / (norm_a * norm_b)
    cos = np.clip(cos, -1.0, 1.0)          # 防止浮点误差导致 |cos| 略大于 1
    return float(np.degrees(np.arccos(cos)))


def np_e3():
    a = np.array([1.00, -1.00])
    b = np.array([1.00, 0.00])

    print("a =", a, " b =", b)
    print(f"1) a @ b = {a @ b:.6f}")
    print(f"   ||a|| = {np.linalg.norm(a):.6f}, ||b|| = {np.linalg.norm(b):.6f}")
    print(f"   cos(theta) = {(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)):.6f}")

    theta = angle_deg(a, b)
    print(f"2) clip 到 [-1,1] 后 arccos，结果 = {theta:.6f} 度")

    print("3) 零向量的边界情况：")
    try:
        angle_deg(np.array([0.0, 0.0]), b)
    except ValueError as err:
        print("   抛出 ValueError:", err)

    print(f"   给定输入的夹角 = {theta:.6f} 度（输入 2 维，输出 1 个标量）")


# ======================================================================
# NP-E4 两个点的距离
# ======================================================================
def np_e4():
    p = np.array([-0.50, 1.20, 2.00])
    q = np.array([0.80, -0.40, 1.10])

    diff = p - q
    show("1) 差向量 p - q", diff)

    # 2) 欧氏距离 = sqrt(sum((p-q)^2))
    euclid = np.sqrt(np.sum(diff ** 2))
    show("2) 欧氏距离", euclid)

    # 3) 曼哈顿距离 = sum(|p-q|)
    manhattan = np.sum(np.abs(diff))
    show("3) 曼哈顿距离", manhattan)

    print(f"   核对 np.linalg.norm(p - q) = {np.linalg.norm(diff):.6f}")


# ======================================================================
# NP-E5 把向量变为单位长度
# ======================================================================
def unit_vector(a):
    """返回与 a 同方向、长度为 1 的新数组；零向量 -> ValueError。"""
    a = np.asarray(a, dtype=float)
    n = np.linalg.norm(a)
    if n == 0.0:
        raise ValueError("零向量无法归一化（没有方向）")
    return a / n          # 除法生成新数组，不改 a


def np_e5():
    a = np.array([-1.20, 0.40, 0.80])
    show("a", a)
    print(f"||a|| = {np.linalg.norm(a):.6f}")

    u = unit_vector(a)
    show("1/2) 单位向量 u = a / ||a||", u)
    print("   原数组未被修改：", np.allclose(a, [-1.20, 0.40, 0.80]))

    print(f"3) u 的长度 = {np.linalg.norm(u):.12f}")
    print("   np.allclose(||u||, 1) =", np.allclose(np.linalg.norm(u), 1.0))

    try:
        unit_vector(np.zeros(3))
    except ValueError as err:
        print("   零向量边界：抛出 ValueError:", err)


# ======================================================================
# NP-E6 沿不同 axis 汇总
# ======================================================================
def np_e6():
    A = np.array([[0.20, -0.50, 1.10],
                  [0.80, 0.40, -0.30]])
    print("A =")
    print(A)

    s0 = A.sum(axis=0)
    m1 = A.mean(axis=1)
    show("1) A.sum(axis=0)", s0)
    show("2) A.mean(axis=1)", m1)
    show("3) A.max()", A.max())

    print("\n   解释：")
    print("   axis=0：沿着第 0 轴（行）压缩掉 -> 每一列求和，结果长度 = 列数 3")
    print("      [0.20+0.80, -0.50+0.40, 1.10-0.30]")
    print("   axis=1：沿着第 1 轴（列）压缩掉 -> 每一行求平均，结果长度 = 行数 2")
    print("      [(0.20-0.50+1.10)/3, (0.80+0.40-0.30)/3]")
    i, j = np.unravel_index(np.argmax(A), A.shape)
    print(f"   整体最大值 {A.max():.6f} 位于 A[{i}][{j}]（第 {i+1} 行第 {j+1} 列）")


# ======================================================================
# NP-E7 一次广播，处理所有行
# ======================================================================
def np_e7():
    X = np.array([[0.80, -0.60, 0.70],
                  [-0.40, 0.50, 0.20]])
    scale = np.array([1.50, 0.50, -1.00])
    bias = np.array([-0.10, 0.20, 0.30])

    # 1) 一条表达式，不写循环
    Y = X * scale + bias
    show("X", X)
    show("scale", scale)
    show("bias", bias)
    show("1/2) Y = X * scale + bias", Y)

    print("\n3) 对齐方式：")
    print("   X 的 shape 是 (2, 3)，scale / bias 的 shape 是 (3,)")
    print("   广播规则从右往左比：3 与 3 匹配，scale 自动复制成 (2, 3)")
    print("   于是每一个标量都作用在同一列（同一特征）上，两行共用同一套参数")
    print("   核对第 1 行手工结果：0.80*1.50-0.10 =", f"{0.80*1.50-0.10:.6f}")


# ======================================================================
# NP-E8 把两张图像组成一个 batch
# ======================================================================
def np_e8():
    a = np.array([[[0.20, -0.40], [0.70, 0.10]],
                  [[0.30, 0.80], [-0.20, 0.50]],
                  [[-0.10, 0.60], [0.40, -0.30]]])
    b = a + 0.10
    print("单张图像按 (C, H, W) 组织 -> a.shape =", a.shape)

    batch = np.stack([a, b], axis=0)
    show("1) batch = np.stack([a, b], axis=0)", batch)
    print(f"   batch.shape = {batch.shape}  -> (B, C, H, W) = (2, 3, 2, 2)")

    blue = batch[:, 2]
    show("2) 蓝色通道 batch[:, 2]", blue)
    print(f"   shape = {blue.shape}  -> (B, H, W)，两张图各一张 2x2 的蓝通道")

    hwc = batch.transpose(0, 2, 3, 1)
    print(f"3) 转为 (B, H, W, C) 后 shape = {hwc.shape}")


# ======================================================================
# NP-M1 批量计算夹角
# ======================================================================
def angles_to(query, X):
    """
    返回 X 每一行与 query 的夹角（度），shape = (n_rows,)。
    - 用 X @ query 一次算出所有行的点乘
    - 用 np.linalg.norm(X, axis=1) 一次算出所有行的长度
    - X 中的零向量行返回 np.nan；query 为零向量则抛 ValueError
    """
    query = np.asarray(query, dtype=float)
    X = np.asarray(X, dtype=float)
    if query.ndim != 1 or X.ndim != 2:
        raise ValueError("query 需为一维，X 需为二维")
    if X.shape[1] != query.shape[0]:
        raise ValueError(f"维度不匹配: X{X.shape} 与 query{query.shape}")

    q_norm = np.linalg.norm(query)
    if q_norm == 0.0:
        raise ValueError("query 是零向量，无法定义夹角")

    dots = X @ query                          # (n,)
    x_norms = np.linalg.norm(X, axis=1)       # (n,) 按行求长度
    valid = x_norms > 0                       # 零向量行的掩码

    cos = np.full(dots.shape, np.nan)         # 先全部填 nan
    with np.errstate(divide="ignore", invalid="ignore"):
        cos[valid] = dots[valid] / (x_norms[valid] * q_norm)
    cos[valid] = np.clip(cos[valid], -1.0, 1.0)

    return np.degrees(np.arccos(cos))         # nan 经 arccos 仍是 nan


def np_m1():
    query = np.array([0.60, -0.80, 0.00])
    X = np.array([[0.30, -0.40, 0.00],
                  [-0.60, 0.80, 0.00],
                  [0.80, 0.60, 0.00],
                  [0.00, 0.00, 0.00]])

    show("query", query)
    print("X =")
    print(X)

    angles = angles_to(query, X)
    show("1) angles_to(query, X)", angles)
    print("   输出 shape 为 (4,)，顺序与 X 的行顺序一一对应")
    print("   query 与 X 都形如 (n,) / (m, n)，矩阵乘 X @ query 得到 (m,)")
    print("   最后一行是零向量 -> 返回 nan（不报错、不参与比较）")

    print("\n3) query 为零向量的边界：")
    try:
        angles_to(np.zeros(3), X)
    except ValueError as err:
        print("   抛出 ValueError:", err)


# ======================================================================
# NP-M2 用余弦相似度做 Top-k 检索
# ======================================================================
def cosine_scores(query, gallery):
    """返回每行的余弦相似度；零向量行返回 np.nan。"""
    query = np.asarray(query, dtype=float)
    gallery = np.asarray(gallery, dtype=float)
    if query.ndim != 1 or gallery.ndim != 2:
        raise ValueError("query 需为一维，gallery 需为二维")
    if gallery.shape[1] != query.shape[0]:
        raise ValueError(f"维度不匹配: gallery{gallery.shape} 与 query{query.shape}")

    q_norm = np.linalg.norm(query)
    if q_norm == 0.0:
        raise ValueError("query 是零向量，无法计算余弦相似度")

    dots = gallery @ query
    g_norms = np.linalg.norm(gallery, axis=1)
    valid = g_norms > 0

    scores = np.full(dots.shape, np.nan)
    scores[valid] = dots[valid] / (g_norms[valid] * q_norm)
    scores[valid] = np.clip(scores[valid], -1.0, 1.0)
    return scores


def top_k_cosine(query, gallery, k):
    """
    返回 (选中行的原始索引, 对应相似度)，按相似度从高到低。
    - 零向量行不参与排序
    - 分数相同保持原行顺序（stable 排序）
    - k 必须为正整数且不超过有效行数
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("k 必须是正整数")
    gallery = np.asarray(gallery, dtype=float)
    scores = cosine_scores(query, gallery)

    valid_idx = np.flatnonzero(~np.isnan(scores))     # 有效行的原始下标
    if k <= 0 or k > valid_idx.size:
        raise ValueError(f"k 需在 1..{valid_idx.size} 之间，收到 {k}")

    valid_scores = scores[valid_idx]
    # 取负号实现降序；kind="stable" 保证同分时按原顺序
    order = np.argsort(-valid_scores, kind="stable")
    picked = valid_idx[order[:k]]
    return picked, scores[picked]


def np_m2():
    query = np.array([0.60, -0.80, 0.00])
    gallery = np.array([[0.60, -0.80, 0.00],
                        [0.30, -0.40, 0.00],
                        [0.80, 0.60, 0.00],
                        [-0.60, 0.80, 0.00],
                        [0.00, 0.00, 0.00]])
    ids = np.array(["img_a", "img_b", "img_c", "img_d", "img_zero"])

    show("query", query)
    print("gallery =")
    print(gallery)
    print("ids =", ids)

    print("\n0) 全部相似度（nan 表示零向量行）:")
    show("   scores", cosine_scores(query, gallery))

    idx, sc = top_k_cosine(query, gallery, k=3)
    show("1) top_k_cosine(k=3) 索引", idx)
    show("   对应相似度", sc)
    show("4) 选中 ID", ids[idx])
    print("   第 1、2 名分数都是 1.0（同分），stable 排序保持了 img_a 在 img_b 之前")

    print("\n5) k 越界的边界：")
    for bad in (0, 6):
        try:
            top_k_cosine(query, gallery, k=bad)
        except ValueError as err:
            print(f"   k={bad} -> ValueError: {err}")


# ======================================================================
# NP-M3 所有点对之间的距离矩阵
# ======================================================================
def np_m3():
    A = np.array([[0.20, -0.40],
                  [0.80, 0.30],
                  [-0.50, 0.70]])
    B = np.array([[0.10, 0.20],
                  [-0.60, -0.20]])
    print("A =")
    print(A)
    print("B =")
    print(B)

    # 1) 广播构造差数组：(3,1,2) - (1,2,2) -> (3,2,2)
    diff = A[:, None, :] - B[None, :, :]
    print(f"\n1) A[:, None, :] shape = {A[:, None, :].shape}")
    print(f"   B[None, :, :] shape = {B[None, :, :].shape}")
    print(f"   差数组 diff shape = {diff.shape}  -> (i, j, 坐标)")
    print("   广播：A 变成 (3,1,2)、B 变成 (1,2,2)，第 0/1 轴上 1 各自被复制，")
    print("   结果 (3,2,2) 正好覆盖 3x2 = 6 个点对，最后那个 2 是坐标维")

    # 2) 沿坐标轴求平方和再开方
    D = np.sqrt(np.sum(diff ** 2, axis=2))
    show("2/3) 距离矩阵 D", D)
    print(f"   D.shape = {D.shape}，D[i][j] 即 A[i] 到 B[j] 的距离")

    nearest = np.argmin(D, axis=1)
    show("3) A 每行最近的 B 行索引", nearest)
    print("   对应最近距离：", np.round(D[np.arange(len(A)), nearest], 6))


# ======================================================================
# NP-M4 用矩阵运算完成一次前向计算
# ======================================================================
def np_m4():
    X = np.array([[0.80, -0.60, 0.70],
                  [-0.40, -0.50, 0.90]])
    W1 = np.array([[-0.10, 0.20, -0.80],
                   [0.30, -0.50, 0.10]])
    b1 = np.array([-0.15, 0.05])
    W2 = np.array([[0.20, -0.60],
                   [-0.80, 0.52]])
    b2 = np.array([0.10, -0.20])

    print("X", X.shape, " W1", W1.shape, " b1", b1.shape)
    print("W2", W2.shape, " b2", b2.shape)
    print("W 按 (输出特征数, 输入特征数) 存储，所以左乘时用转置 W.T")

    # 1) 线性：X(2,3) @ W1.T(3,2) -> (2,2)，再加偏置
    Z = X @ W1.T + b1
    show("\n1) Z = X @ W1.T + b1", Z)

    # 2) 逐元素 ReLU
    H = np.maximum(Z, 0.0)
    show("2) H = np.maximum(Z, 0.0)", H)

    # 3) 再线性：H(2,2) @ W2.T(2,2) -> (2,2)
    Y = H @ W2.T + b2
    show("3) Y = H @ W2.T + b2", Y)

    print(f"\n4) shape 链：X{X.shape} -> Z{Z.shape} -> H{H.shape} -> Y{Y.shape}")
    print("   全程无样本循环；参数数组未被修改")


# ======================================================================
# 额外：边界输入速查（讲评准备用）
# ======================================================================
def edge_cases():
    print("\n--- 边界输入速查 ---")
    print("1) 空/单行矩阵：angles_to(query, X[:1]) 应返回 shape (1,)")
    q = np.array([0.60, -0.80, 0.00])
    one = np.array([[0.30, -0.40, 0.00]])
    print("   ->", angles_to(q, one), "shape =", angles_to(q, one).shape)

    print("2) 反向向量：夹角应为 180 度")
    print("   ->", f"{angle_deg(np.array([1.0, 0.0]), np.array([-1.0, 0.0])):.6f}")

    print("3) 相同向量：夹角应为 0 度")
    print("   ->", f"{angle_deg(np.array([1.0, 0.0]), np.array([1.0, 0.0])):.6f}")

    print("4) 正交向量：夹角应为 90 度")
    print("   ->", f"{angle_deg(np.array([1.0, 0.0]), np.array([0.0, 1.0])):.6f}")

    print("5) 单元素数组：A = np.array([[5.0]]) 的 sum(axis=0) shape 为 (1,)")
    sin = np.array([[5.0]])
    print("   ->", sin.sum(axis=0), "shape =", sin.sum(axis=0).shape)


def main():
    title("NP-E1 数组、形状与切片")
    np_e1()

    title("NP-E2 逐元素乘法、点乘与长度")
    np_e2()

    title("NP-E3 两个向量的夹角")
    np_e3()

    title("NP-E4 两个点的距离")
    np_e4()

    title("NP-E5 把向量变为单位长度")
    np_e5()

    title("NP-E6 沿不同 axis 汇总")
    np_e6()

    title("NP-E7 一次广播，处理所有行")
    np_e7()

    title("NP-E8 把两张图像组成一个 batch")
    np_e8()

    title("NP-M1 批量计算夹角")
    np_m1()

    title("NP-M2 用余弦相似度做 Top-k 检索")
    np_m2()

    title("NP-M3 所有点对之间的距离矩阵")
    np_m3()

    title("NP-M4 用矩阵运算完成一次前向计算")
    np_m4()

    title("边界输入速查")
    edge_cases()


if __name__ == "__main__":
    main()
