# Project 1 学习手册 · 用 NumPy 复现 ViT 图像编码器

> 对应课件：`01-NumPy-Arrays-and-ViT-Forward.pptx`（76 页）
> 对应作业：`project1-numpy-student/`（只改 `image_encoder.py`，填 6 个 TODO）

这份文档解决三件事：

1. **看懂作业**——到底要交什么、检查怎么过、怎么算分（第 1 部分）
2. **看懂课件**——76 页 PPT 逐板块讲透，每一步都追着 shape 走（第 2 部分）
3. **知道从哪下手**——6 个 TODO 逐个给提示卡、易错点和验收命令（第 3、4 部分）

配套文档：

| 文件 | 内容 |
|---|---|
| `README.md` | 本手册（作业 + 课件精讲 + TODO 提示） |
| `练习12题详解.md` | 热身练习 12 题的原文与逐题详解（`numpy_homework.py` 是其答案代码） |
| `week7课件精讲.md` | 第 7 周《GitHub 与 Overleaf》课件精讲（提交要用） |
| `numpy作业.py` | 你自己的练习尝试（NP-E1），保持不动 |
| `project1-numpy-student/` | 官方学生包，作业在这里做 |
| `lecture-demos/` | 课堂示例脚本（`01_numpy_basics.py` 等） |
| `Experiment-Report-Template-SZU/` | 老师的 LaTeX 报告模板（`main.tex` + `figs/`） |

---

# 第 1 部分 作业要求详解

## 1.1 这份作业到底要做什么

一句话：**老师给了训练好的 OpenCLIP ViT-B-32 视觉塔全部参数（152 个张量）和 10 张测试图片，你用纯 NumPy 把"一张图片 → 一个 512 维向量"的前向计算写出来，数值必须和参考输出对上。**

- 不需要训练、不需要 GPU、不需要 PyTorch，只要 NumPy + Pillow
- 你写的 `ImageEncoder` 之后会**原封不动**搬进期末大项目（图文多模态检索）当图像塔
- 只允许改 `image_encoder.py` 一个文件，共 6 个 TODO

### 全程 shape 流水线（这张表就是整份作业的地图）

```
图片 x                    [B, 3, 224, 224]     ← 预处理已提供
  │ patchify              (TODO-01)
  ▼ patches               [B, 49, 3072]        ← 7×7=49 块，每块 3×32×32 个值
  │ @ patch_weight.T      (TODO-05a，无 bias)
  ▼ patch_tokens          [B, 49, 768]         ← D = 768
  │ 拼上 CLS + 加位置      (TODO-05b)
  ▼ x                     [B, 50, 768]         ← N = 49+1 = 50 个 token
  │ pre_norm              (已提供)
  ▼
  │ 12 × TransformerBlock（结构相同，参数各自独立）
  │   n1 = Norm1(x)                              (已提供)
  │   qkv = n1 @ qkv_w.T + b  → [B,50,2304]      (已提供)
  │   切成 Q,K,V            → [B,50,768] ×3      (已提供)
  │   split_heads           → [B,12,50,64]       (TODO-02)
  │   scores = Q@Kᵀ/√64     → [B,12,50,50]       (TODO-03a)
  │   softmax               (已提供)
  │   context = A@V         → [B,12,50,64]
  │   合并 12 个头           → [B,50,768]
  │   @ out_proj            (TODO-03c)
  │   residual = x + update                      (TODO-04a)
  │   u = Norm2(residual)                        (已提供)
  │   MLP: fc1→GELU→fc2     → [B,50,3072]→[B,50,768]  (TODO-04b)
  │   output = residual + mlp
  ▼ last_tokens            [B, 50, 768]
  │ 取第 0 行 CLS          (TODO-06)
  ▼ cls                    [B, 768]
  │ post_norm
  ▼                        [B, 768]
  │ @ visual.proj          ← 已是 [768,512]，不转置、无 bias
  ▼                        [B, 512]
  │ unit_vector            (已提供)
  ▼ output_tokens          [B, 512]             ← 最终交付
```

记号约定（课件 slide 50）：**B** 图片张数、**N** token 行数（50）、**D** 每个 token 的特征数（768）、**H**（或 n_head）注意力头数（12）、**d_head** 每个头的维度（64）、**M** MLP 中间宽度（3072）。

## 1.2 六个 TODO 总表

| TODO | 方法 | 输入 → 输出 | 一句话任务 |
|---|---|---|---|
| 01 | `ImageEncoder.patchify` | `[B,3,224,224]` → `[B,49,3072]` | 把图片切成 49 个 patch 行，块内保持 C,H,W 顺序 |
| 02 | `MultiHeadAttention.split_heads` | `[B,N,D]` → `[B,H,N,D/H]` | 把特征轴拆成 12 个头（需自行研究头组织） |
| 03 | `MultiHeadAttention.__call__` | 见下 | scores、加权求和、合并头、输出投影 |
| 04 | `TransformerBlock.__call__` | `[B,N,D]` → `[B,N,D]` | 两条残差 + MLP |
| 05 | `ImageEncoder.embed` | `[B,3,224,224]` → `[B,50,768]` | patch 投影、拼 CLS、加位置 |
| 06 | `ImageEncoder.__call__` | `[B,50,768]` → `[B,512]` | 取 CLS、post-Norm、投影、单位化 |

## 1.3 七条模型约定（写之前必读，全来自学生包 README）

1. **Patch 是 32×32**：224/32 = 7，所以 7×7 = 49 个 patch，每块拍平 `3*32*32 = 3072` 个值。
2. **Patch 投影没有 bias**：只有 `patches @ patch_weight.T` 一步。
3. **Attention/MLP 的权重按 `[out,in]` 存**：左乘一律加 `.T`，即 `x @ W.T + b`。
4. **`visual.proj` 例外**：它已经是 `[768,512]`（即 `[in,out]`），**不转置、没有 bias**。
5. **激活和注意力系数转换用提供的**：`helpers.py` 里的 `GELU`、`attention_weights`、`LayerNorm`（含 `eps=1e-5`）、`unit_vector`，不要自己另写一份。
6. **图像 block 有两条残差加法**，没有 LayerScale 参数。
7. **不做任何参数更新**，`config.json` 描述的是这个固定 checkpoint，改配置数字不会变成另一个模型。

## 1.4 怎么验证自己写对了

所有命令在 `project1-numpy-student/` 目录下运行：

```bash
python -m pip install -r requirements.txt   # numpy、pillow、imageio

# 起步检查（学生包未改动时就该全过）
python check.py --stage assets
python check.py --stage preprocess

# 按实现顺序逐级检查（每级只报第一个没过的 TODO）
python check.py --stage patches        # TODO-01
python check.py --stage tokens         # TODO-05
python check.py --stage block0         # TODO-02/03/04 的第一个 block
python check.py --stage last_tokens    # 12 个 block 全部
python check.py --stage output         # TODO-06
python check.py --stage batch          # 乱序/重复图片也能算对
python check.py --stage all            # 总验收

# 生成提交用的输出文件
python run.py --output output_tokens.npz
```

**容差规则**：`abs(实际 - 参考) ≤ atol + rtol × abs(参考)`

| 阶段 | atol | rtol |
|---|---|---|
| preprocess | 2e-6 | — |
| 内部 token（patches/tokens/block0/last_tokens） | 2e-4 | 2e-4 |
| 最终 output_tokens | 2e-5 | 2e-4 |

check.py 会打印每阶段的**最大绝对误差和平均绝对误差**——这两个数直接写进报告，就是"结果与证据"。

调试方法（README 原话的翻译）：拿 `assets/reference_outputs.npz` 里的中间数组（`last_tokens` 形状 `(10,50,768)`、`output_tokens` 形状 `(10,512)`）逐阶段对比，**找到第一个开始出现偏差的阶段**，在那里定位 bug。注意：这些参考数组只能用来 debug，不许喂给自己的编码器当输入，也不许直接返回缓存结果——你的实现必须对任意图片、任意 batch 大小都成立。

还有一个迷你演算程序，把 slide 46-66 的算术用小矩阵走了一遍，强烈建议先跑：

```bash
python transformer_steps.py
```

它用 3 个 token、4 个特征、1 个头、6 个 MLP 中间值演示：行均值、去中心、方差、Norm 的 scale/bias、q/k/v、scores、系数、加权和、注意力投影、两条残差和 MLP。**它故意没做多头组织**（那是 TODO-02 的独立研究部分），也不使用训练好的参数。

## 1.5 提交要求

两处来源：`项目提交要求.png`（老师的要求页）+ 学生包 README 的 Submission 节。

**要交的三样**：

| 材料 | 要求 |
|---|---|
| `image_encoder.py` | 6 个 TODO 全部完成，`check.py --stage all` 通过 |
| **README**（仓库里） | 说明怎么运行；注释充分（Sufficient annotations and comments） |
| **实验报告 PDF** | 用 Overleaf（LaTeX）写，**少于 5 页**，精简（Less is more） |

**报告里必须出现**（学生包 README 原文要求）：

1. **每个阶段的数值误差**（最大/平均绝对误差表——check.py 直接给你）
2. **解释一个 shape 变化**（比如 patchify 的六轴 reshape+transpose）
3. **解释你修过的一个 bug**（现象 → 定位过程 → 原因 → 修法）
4. **本人贡献与使用的辅助**（课程要求 "Highlight Your Parts in RED"——用红色标出自己的关键工作，AI 等辅助也要注明）
5. GitHub 链接（代码放 GitHub，报告用 Overleaf 写，两者是计分项）

**不要提交**：`weights/image_encoder.npz`（152 个张量的大文件）和 `reference_outputs.npz`（参考数据）。

**计分**（与 week7 课件一致）：单项 =（代码 30 + 报告 70）× 面试系数 0.6~1.0。报告占 70 分，所以"误差表 + shape 讲解 + bug 复盘"这三块别省。

---

# 第 2 部分 课件精讲（76 页）

> **本部分是速览版。逐页精讲（每页"在问什么 / 数字怎么来 / 易错 / 对应哪行代码"，含 76 页一句话索引、自测题与答案）见同目录 `NumPy课件精讲.md`。**

> 课件回答 5 个问题（slide 2），前 4 问是地基，第 5 问就是这份作业：
> ① 数据怎么表示 ② 数据怎么组织 ③ 数据怎么算 ④ 图片怎么变成向量 ⑤ 你要完成哪些部分
>
> 课件的教学法（备注里写明的）：**先说出每个轴是什么 → 预测输出 shape → 再看算术**。看 PPT 时照这个顺序读，会顺很多。

## 2.0 课件地图

| 板块 | 页码 | 学完能做什么 |
|---|---|---|
| 1 Represent data | 3–8 | 建数组、读 shape/ndim/size、把图片看成通道堆叠 |
| 2 Arrange data | 9–18 | 索引/切片、reshape、transpose、加轴、拼接、组 batch |
| 3 Calculate | 19–28 | 逐元素运算、点乘、矩阵乘、按轴求和、广播、keepdims |
| 4 Follow an image model | 29–38 | 读图、预处理、看懂线性层和 MLP 的前向 |
| 5 图像编码器（作业本体） | 39–76 | patch → token → 12 个 block → 512 维输出 |

贯穿全课的玩具数据：`[0.80, -0.60, 0.70, -0.40, -0.50, 0.90]`（两个 (2,3) 网格的展开）。**课件里所有两位小数都只是显示值，计算必须用全精度**——这句话在备注里反复出现，因为作业容差是 2e-5 量级，中途四舍五入必挂。

## 2.1 板块一：数据表示（slide 3–8）

**核心思想：一个数值在数组里有"位置"，shape 就是"每个方向有多长"。**

| 页 | 知识点 | 要点 |
|---|---|---|
| 3–5 | `np.array`、shape/ndim/size | `shape` 各轴长度；`ndim` 轴数（=len(shape)）；`size` 总元素数。6 个值排成 (2,3)，ndim=2、size=6 |
| 6 | 随机创建 | `rng = np.random.default_rng(7)` 固定种子可复现；`rng.uniform(-1,1,(2,3))`、`np.zeros/ones/linspace` |
| 7 | RGB 图像 = 三层网格 | `(C,H,W)`：红绿蓝各是一张 (H,W) 网格叠起来；`img[c,h,w]` 取一个值；`img[0]` 取整层，shape 变 (H,W) |
| 8 | 0-D 到 5-D | 每加一个轴，就把前面的整体"打包"一层：标量 `()` → 序列 `(l,)` → 网格 `(h,w)` → 图像 `(c,h,w)` → 图像 batch `(b,c,h,w)` → 视频 `(b,t,c,h,w)` |

**课件反复强调的轴约定**：图像用 **CHW**、图像批用 **BCHW**；只有读文件/存文件的瞬间（imageio）是 HWC，要显式 transpose。这直接对应作业输入 `[B,3,224,224]`。

## 2.2 板块二：数据组织（slide 9–18）

**核心思想：每做一次操作，先报出前后 shape。**

### 索引 vs 切片（slide 10）

```python
a[1]     # 整数索引 → 消掉行轴，shape (3,)
a[1:2]   # 切片     → 保留行轴，shape (1,3)
a[:,1:3] # 切片     → 保留行轴，取两列
```
一句话：**整数索引降维，切片保维；切片含头不含尾。**

### reshape（slide 11）：**保持顺序，只改分组**

`[0.80,-0.60,0.70,-0.40,-0.50,0.90]` 可以 reshape 成 (2,3) 也可以成 (3,2)——元素顺序不变，size 不变。reshape **不会**把元素挪到"该在的位置"，它只是重新读轴。这是 TODO-01 的基础。

### transpose（slide 12）：**整轴交换**

`.T` 后 `a.T[j,i] == a[i,j]`，(2,3) 变 (3,2)。reshape 是"重新分组"，transpose 是"轴换位置"——两个都改变 shape 但含义完全不同，这是最容易被混的一对。

### 加一个长度为 1 的轴（slide 13）

```python
v[None, :]   # (3,) → (1,3)  行向量
v[:, None]   # (3,) → (3,1)  列向量
```
一维向量本身没有"行/列"之分，`None` 用来造出这个轴——广播和 TODO-05 的位置加法都靠它。

### concatenate vs stack（slide 14–18，本板块的考点）

| 操作 | 作用 | 例子 |
|---|---|---|
| `concatenate([a,b], axis=0)` | **延长已有轴** | (2,2)+(1,2) → (3,2) |
| `concatenate([a,b], axis=1)` | 延长列轴（其余轴长度必须一致） | (2,2)+(2,1) → (2,3) |
| `np.stack([a,b], axis=0)` | **新建一个轴** | 两张 (3,2,3) 图 → (2,3,2,3) 的 batch |

slide 18 的 checkpoint 值得背下来：两张 `(3,2,3)` 的图，
- `stack` → `(2,3,2,3)`：新 batch 轴，2 张图
- `concatenate` → `(6,2,3)`：把通道轴接长成 6 层（多半不是你想要的）

**结论：单图组 batch 用 stack；已有 batch 合并用 concatenate。**

## 2.3 板块三：计算（slide 19–28）

### 向量化（slide 20）

`a*0.50+0.10` 一条表达式替代循环，逐位置应用同一条规则，shape 不变。`np.abs(a)`、`a**2` 同理。

### 逐元素乘 vs 点乘（slide 21–22）

```python
a * b   # 对应位置相乘 → (3,) 还是数组
a @ b   # 乘完再求和   → () 是一个数
```
课件例：`a*b = [-0.32, 0.30, 0.63]`，`a@b = -0.32+0.30+0.63 = 0.61`。
这就是热身练习 NP-E2 的核心，也是一切线性层的原子操作。

### 矩阵乘（slide 23–24）

- `(3,) @ (3,2)` → `(2,)`：每一列贡献一个输出特征
- `(h,k) @ (k,w)` → `(h,w)`：**内维 k 必须相等，外维保留**
- 输出格 `Y[i,j]` = A 的第 i 行与 W 的第 j 列的点乘

### sum 与 axis（slide 25）

**`axis=k` 就是"把第 k 维消掉"**：
- `a.sum(axis=0)`：(2,3) → (3,)，每列求和
- `a.sum(axis=1)`：(2,3) → (2,)，每行求和

### 广播（slide 26）

**shape 从右往左比，每一对要么相等要么有一个是 1**：
- `(2,3) + (3,)`：行向量给每列一个偏移
- `(2,3) + (2,1)`：列向量给每行一个偏移

### keepdims（slide 27，LayerNorm 的钥匙）

```python
m = a.mean(axis=1, keepdims=True)   # (2,3) → (2,1)
a - m                                # 每行减自己的均值
```
`keepdims=True` 让被消的轴保留成长度 1，这样减法能广播回原 shape。**slide 47 的 LayerNorm 第一行就是这个套路**：`x.mean(axis=-1, keepdims=True)` 把 (B,N,D) 变 (B,N,1)。

slide 28 的 checkpoint 值得自测：`A(2,3)*A` → (2,3)；`A.sum(axis=0)` → (3,)；`A@W(3,2)` → (2,2)；`A@W+b(2,)` → (2,2)（b 广播到两行）。

## 2.4 板块四：从图片到模型（slide 29–38）

| 页 | 知识点 | 对作业的意义 |
|---|---|---|
| 30 | imageio 读文件是 **HWC**，算之前 `transpose(2,0,1)` 转 **CHW**，存图再转回去 | 理解为什么预处理后是 (3,224,224) |
| 31 | 一个像素 = 同一 (h,w) 位置上的三个通道值：`img[:,90,140]` → `[131 89 64]` | patch 拍平时"块内保持 C,H,W"的原因 |
| 32 | 先 `astype(np.float32)` 再算术（防 uint8 回绕）、`clip` 限幅；切片是**视图**（改它 = 改原图），`.copy()` 才独立 | 作业输入是 float32 的原因；视图/副本区别 |
| 33 | 图像编辑全是数组操作：通道切片、空间裁剪、`gain[:,None,None]` 乘通道 | `(3,1,1)` 广播形状的实战 |
| 34 | 模型预处理：`/255` → `(x-mean)/std`，mean/std 是 OpenCLIP 随模型提供的常数，扩成 (3,1,1) | `preprocess_image.py` 已提供，别自己造 |
| 35 | 模型 = `y = fθ(x)`：x 是随图变的输入，θ={W1,b1,W2,b2,…} 是**已训练好、固定不动**的参数 | 全篇最重要的一句话：你只实现前向，不学习 |
| 36–38 | 线性层 `z = x @ W + b`（(1,3)@(3,2)+(1,2)→(1,2)）；MLP = linear→ReLU→linear | TODO-04 的迷你版；作业里把 ReLU 换成提供的 GELU，W 换成 `W.T`（因为按 [out,in] 存储） |

## 2.5 板块五：图像编码器（slide 39–76，作业本体）

### 39–45：从图片到 token

**slide 39 总览**（右侧竖条会在每一页重复出现，标出"你现在讲到哪一步"）：

```
[B,3,224,224] → patch 投影 → CLS+位置 → pre-Norm → 12 blocks
→ CLS 读出 + 投影 → [B,512]
```

- **slide 40–41 图像准备**：resize（短边到 224）→ 中心裁剪 → 归一化 → 组 batch。自然图片尺寸比例各异，但张量形状必须固定，所以需要"形状对齐"。这一段已提供（`preprocess_image.py`）。
- **slide 42–43 patchify（TODO-01）**：224 = 7×32，所以图片是 7×7=49 个 patch；每 patch 拍平 `3×32×32=3072` 个值，**按从左到右、从上到下访问 patch，patch 内部保持 C,H,W 顺序**。slide 43 给出完整轴变换链：

```
(B,3,224,224) → reshape → (B,3,7,32,7,32) → transpose → (B,7,7,3,32,32) → reshape → (B,49,3072)
```
  解读：先把 H、W 各拆成 (7,32) 两段（"7 个 patch、每 patch 32 像素"），此时轴顺序是 B,C,pH,pH内,pW,pW内；再把轴排成 B,pH,pW,C,pH内,pW内；最后把后三维合成 3072。
- **slide 44 patch 投影（TODO-05a）**：`patches @ patch_weight.T`，`[B,49,3072] @ [3072,768] → [B,49,768]`。一个共享矩阵把每个 patch 行变成 768 维 token，**无 bias**。
- **slide 45 CLS + 位置（TODO-05b）**：在 token 轴（axis=1）最前面拼一个可学习的 CLS 行 → 50 行；再加位置嵌入 `self.position`（(50,768)，对 B 广播）。**N 从这里开始是 50（49 patch + 1 CLS），CLS 永远在第 0 行。**

### 46–52：Norm 与 block 骨架

- **slide 46**：LayerNorm 的直觉——把每行数据"整理分布"再缩放平移（课件自己标注这是"不完全严格的直觉解释"）。
- **slide 47–49 Norm 三步**（在 `helpers.py` 已提供，但要能读懂）：

```python
mean        = x.mean(axis=-1, keepdims=True)          # [B,N,1]
centered    = x - mean                                 # [B,N,D]
variance    = (centered*centered).mean(-1, keepdims=True)
standardized= centered / np.sqrt(variance + 1e-5)
n1          = standardized * weight + bias             # [D] 广播到 [B,N,D]
```
  均值/方差都是**沿最后一维 D** 算的：每张图、每个 token 行各自归一化；scale/bias 是可学习参数，广播复用。
- **slide 50–51**：`[B,N,D]` 读法——B 张图、每张 N 行 token、每行 D 个特征；线性参数统一存 `weight:[out,in]`、`bias:[out]`。Norm1 和 Norm2 **计算图相同、参数各自独立**（两套 weight/bias）。
- **slide 52 block 总公式**（背下来）：

```python
residual = x + Attention(Norm1(x))
output   = residual + MLP(Norm2(residual))
```
  block 收发都是 `[B,N,D]`，所以 12 个 block 可以直接串联。先存住 x（shortcut）再算注意力，残差加法用保存的原图。

### 53–59：注意力内部（TODO-02/03 的全部线索）

slide 53 右侧有一个固定的小流程图，8 步：

```
QKV linear → 切 Q/K/V → 组织轴 → Q·Kᵀ/√d_head → softmax → A@V → 合并头 → 输出线性
```

| 页 | 步骤 | 公式与 shape |
|---|---|---|
| 53 | QKV 投影（已提供） | `qkv = n1 @ qkv_weight.T + qkv_bias`，`[B,N,D]@[D,3D]→[B,N,3D]` |
| 54 | 切三段（已提供） | `q = qkv[...,:D]`、`k = qkv[...,D:2D]`、`v = qkv[...,2D:]`，各 `[B,N,D]` |
| 55 | **split_heads（TODO-02）** | `[B,N,D]` → `[B,n_head,N,d_head]`，`n_head×d_head = D`（12×64=768）。**课件故意不讲**，让你独立研究；提示见第 3 部分 |
| 56 | K 转置 | `k_T = k.swapaxes(-1,-2)`：只换最后两轴，B 和头轴不动，`[B,H,N,64]→[B,H,64,N]` |
| 57 | **scores（TODO-03a）** | `scores = (q @ k.swapaxes(-1,-2)) / np.sqrt(head_dim)` → `[B,H,N,N]`。每个 query 和每个 key 都比一次，N×N 是"注意力表" |
| 58 | **加权和（TODO-03b 前半）** | `coefficients = attention_weights(scores)`（softmax 已提供）、`context = coefficients @ v` → `[B,H,N,64]`。A 的每一行是 50 个权重，对 V 的 50 行做加权平均 |
| 59 | **合并 + 输出投影（TODO-03b/03c）** | 先把 12 个头拼回 `[B,N,768]`，再 `merged @ out_weight.T + out_bias` → `[B,N,D]` |

### 60–66：残差与 MLP（TODO-04）

| 页 | 公式 | shape |
|---|---|---|
| 60 | `residual = x + update`（第一条残差，x 是进 block 的原件） | (B,N,D)+(B,N,D) |
| 61 | `u = Norm2(residual)`（residual 继续留着等第二条残差） | [B,N,D] |
| 62 | `hidden = u @ fc1_weight.T + fc1_bias` | [B,N,768]@[768,3072]→[B,N,3072]，M=4D |
| 63 | `activated = GELU(hidden)`（已提供精确版） | [B,N,3072]，逐位置，不变 shape |
| 64 | `mlp_update = activated @ fc2_weight.T + fc2_bias` | [B,N,3072]@[3072,768]→[B,N,768]，把宽度变回 D 好做残差 |
| 65 | `output = residual + mlp_update`（第二条残差） | [B,N,D] |
| 66 | 12 个 block 串联：`x = self.pre_norm(self.embed(images)); for block in self.blocks: x = block(x)` | 全程 [B,N,D]，CLS 和 patch 行始终一起走 |

### 67–73：输出头与验收（TODO-06）

| 页 | 步骤 | 代码 |
|---|---|---|
| 67 | 取每张图的第 0 行 | `cls = tokens[:, 0, :]`，`[B,50,768]→[B,768]` |
| 68 | post-Norm（独立参数的 LayerNorm） | `normalized_cls = self.post_norm(cls)` |
| 69 | 输出投影 | `output = normalized_cls @ self.projection`，`[B,768]@[768,512]→[B,512]`。**proj 已是 [in,out]，不转置、无 bias** |
| 70 | 单位化 | `output_tokens = unit_vector(output)`（已提供）：每行除以自身长度，长度变 1，shape 不变 |
| 71 | 全链总结 | `tokens → cls → post_norm → @projection → unit_vector`，`last_tokens` 是全部 50 行、`output_tokens` 是每图一行 |
| 72 | 六个任务分布图 | 输入侧 01/05、Transformer 侧 02/03/04、输出侧 06 |
| 73 | 数值检查 | 六个 stage 的含义与容差（见第 1.4 节） |

### 74–76：为终项目铺路

- **74**：写好的 `image_encoder.py` + `helpers.py` 直接拷进终项目当图像塔，接口不变
- **75**：文本侧是"查表"——`token_table[ids]`，一个 token ID 取出一行数值
- **76**：图像塔和文本塔共用同一个 `TransformerBlock` 类，只是 θ、D、n_head 不同；文本塔多了 causal 开关（`causal=True`，代码里已预留）

## 2.6 看懂 PPT 的三个阅读技巧（从课件备注里提炼）

1. **每页先找"蓝色区域"**：蓝色标记"本页计算发生在哪"，其余是输入/输出。
2. **右侧竖条是进度条**：图像编码器的 6 个阶段，每页都高亮当前讲到哪。
3. **两位小数是显示值**：真实计算全精度。课件备注原话：`Compute at full precision and round only for display`——你的实现里不要出现任何 `round`。

---

# 第 3 部分 六个 TODO 提示卡

> 按依赖顺序：01 → 05 → 02 → 03 → 04 → 06。每张卡：契约 / 线索 / 易错点 / 验收。
> 注意课程要求注明 AI 辅助；下面的提示是"怎么想"，关键的多头组织（02）请按课件要求独立研究后再写。

## TODO-01 `patchify`

- **契约**：`[B,3,224,224]` → `[B,49,3072]`，patch 从左到右、从上到下，块内 C,H,W 顺序。
- **线索**：slide 43 给了完整链——`reshape(B,3,7,32,7,32)` → `transpose` 成 `(B,7,7,3,32,32)` → `reshape(B,49,3072)`。
- **怎么想**：H=224 拆成"7 个 patch × 每 patch 32"，W 同理；然后把"patch 网格两轴"提到前面、"C 和像素两轴"放到后面；最后三个小轴合成 3072。transpose 的目标是 `(0, 2, 4, 1, 3, 5)`——自己对着六个轴名推一遍，别抄。
- **易错点**：直接 `reshape(B,49,3072)`（patch 内顺序变 HWC）；transpose 后忘了再 reshape。
- **验收**：`python check.py --stage patches`

## TODO-05 `embed`

- **契约**：`[B,3,224,224]` → `[B,50,768]`。
- **线索**：slide 44–45。`patches @ self.patch_weight.T`（无 bias）→ `np.concatenate([cls, patch_tokens], axis=1)` → `+ self.position`。
- **易错点**：给 patch 投影加了 bias（约定 2 说不加）；CLS 拼到 axis=0（要拼 token 轴 axis=1）；忘加位置；`self.position` 形状 (50,768) 直接加即可（对 B 广播）。
- **验收**：`python check.py --stage tokens`

## TODO-02 `split_heads`

- **契约**：`[B,N,D]` → `[B,H,N,D/H]`（768 拆成 12×64），**保持 token 顺序**。
- **线索**：课件 slide 55 只给契约，头组织要求独立研究。思考路径：
  1. D 轴本质是"12 个头 × 每头 64 维"，所以先 `reshape(B, N, H, 64)`；
  2. 此时头轴在第 2 位，目标要它在第 1 位（N 之前），交换轴 1 和 2；
  3. 对比 TODO-01：这里是"先 reshape 再 transpose"，方向和 patchify 相反。
- **易错点**：`reshape(B, H, N, 64)`——这会把不同 token 的特征混进同一个头；忘了 transpose；用了 `.T`（四维数组 `.T` 会把所有轴倒序，B 都跑到最后）。
- **自查**：拿一个小数组 `(1,3,4)` reshape 成 `(1,3,2,2)` 再换轴，手动验证 `out[0,h,n,:] == x[0,n,h*2:(h+1)*2]`。
- **验收**：`--stage block0` 会用到它（和 03/04 一起）。

## TODO-03 `MultiHeadAttention.__call__`（三小步）

- **03a scores**：`scores = (q @ k.swapaxes(-1,-2)) / np.sqrt(self.head_dim)` → `[B,H,N,N]`。易错：忘了除 `sqrt(64)=8`；`swapaxes` 错成交换 0/1 轴（B、H 必须原地不动）。
- **03b 加权 + 合并**：`context = coefficients @ v`（已提供 softmax 在前一行）→ `[B,H,N,64]`；然后把头拼回 `[B,N,768]`——这是 TODO-02 的**逆操作**：先 `transpose` 换回 `(B,N,H,64)`，再 `reshape(B,N,D)`。
- **03c 输出投影**：`merged @ self.out_weight.T + self.out_bias` → `[B,N,D]`。权重按 `[out,in]` 存，所以 `.T`。
- **注意**：代码里 `causal` 分支和 `attention_weights` 已提供，图片路径 `causal=False`，别动。
- **验收**：`python check.py --stage block0`

## TODO-04 `TransformerBlock.__call__`

- **04a**：`residual = x + update`。变量名必须叫 `residual`（后文 `u = self.norm2(residual)` 在用它）。
- **04b**：`hidden = u @ self.fc1_weight.T + self.fc1_bias`（→ `[B,N,3072]`）→ `activated = self.gelu(hidden)`（已提供）→ `mlp_update = activated @ self.fc2_weight.T + self.fc2_bias`（→ `[B,N,768]`）→ `return residual + mlp_update`。
- **易错点**：第二条残差加的是 `residual` 而不是 `u`（`u` 只喂 MLP）；fc1/fc2 的 `.T` 忘掉；GELU 自己写了一个近似版（必须用提供的精确版，容差 2e-4 会咬人）。
- **验收**：`--stage block0`；12 块全对后 `--stage last_tokens`

## TODO-06 `ImageEncoder.__call__`

- **契约**：`[B,50,768]` → `[B,512]`。
- **线索**：slide 71 的四行就是答案骨架——`cls = tokens[:, 0, :]` → `normalized_cls = self.post_norm(cls)` → `output = normalized_cls @ self.projection`（无 `.T`、无 bias）→ 最后已有的一行 `return unit_vector(output)`。
- **易错点**：取 CLS 用了 `tokens[0]`（那是取第 0 张图的全部 50 行）；给 proj 加了 `.T`；把 `last_tokens` 当输出（`last_tokens` 是内部 50 行序列，输出是每图一行 512）。
- **验收**：`python check.py --stage output`，然后 `--stage all` 全绿。

---

# 第 4 部分 12 道练习题 ↔ 6 个 TODO 的对应

热身题不是随便出的，每道题都在给某个 TODO 铺路（详解见 `练习12题详解.md`）：

| 练习 | 练的是 | 用在哪 |
|---|---|---|
| NP-E1 切片/降维 | 整数索引降维、切片保维 | TODO-06 取 CLS 行 `tokens[:,0,:]` |
| NP-E2 `*` vs `@` | 逐元素 vs 点乘 | slide 21–22，一切 `x @ W.T + b` 的原子 |
| NP-E3 夹角 + clip | 数值稳定性意识 | softmax/归一化前先防越界的同款思维 |
| NP-E5 单位向量 | 除以自身长度 | slide 70 `unit_vector`（已提供，但要懂它） |
| NP-E6 sum axis | axis 消维 | LayerNorm 的 `mean(axis=-1, keepdims=True)` |
| NP-E7 广播 | (2,3)*(3,)+(2,3)+(2,1) | Norm 的 `*weight+bias`（slide 48）、位置加法 |
| NP-E8 stack/transpose | 组 batch、换轴 | TODO-01 patchify 的 transpose、TODO-02 的头组织 |
| NP-M1 批量矩阵乘 | `X @ query` 一次算全部 | "一次处理整个 batch"，全部 TODO 的思想 |
| NP-M2 stable 排序 | 数值细节可复现 | 报告里"误差从哪来"的素材 |
| NP-M3 广播差数组 | 进阶广播 | slide 33 的 `gain[:,None,None]` 同款 |
| NP-M4 两层前向 | linear→激活→linear | **TODO-04b 的 MLP 完全同构**（ReLU 换 GELU、W 换 W.T、加残差） |

---

# 第 5 部分 建议上手顺序

1. **跑通环境**（30 分钟）：装依赖 → `check.py --stage assets`、`--stage preprocess` 全过。这两级不用写代码，过不了说明环境有问题。
2. **跑 `python transformer_steps.py`**：对着 slide 46–66 看小矩阵演算，把 Norm/attention/MLP 的每一步 shape 在纸上标出来。
3. **做 TODO-01**（提示卡在前面）：`--stage patches` 过。
4. **做 TODO-05**：`--stage tokens` 过。到这里你已经能把图片变成 50 个 token 了。
5. **做 TODO-02、03、04**：`--stage block0` 过。这是最难的一关，卡住时用参考 `last_tokens` 对比第 0 个 block 的输出，确认错在 attention 还是 MLP。
6. **做 TODO-06**：`--stage output`、`--stage batch`、`--stage all` 全绿。
7. **`python run.py --output output_tokens.npz`** 生成结果文件。
8. **写报告**（Overleaf，<5 页）：误差表（每阶段 max/mean abs error，check.py 会打印）、讲清一个 shape 变化（推荐 patchify 的六轴变换）、复盘一个 bug、红色标出自己的贡献、附 GitHub 链接。模板在 `Experiment-Report-Template-SZU/`。
9. **交三样**：完成的 `image_encoder.py`、仓库 README（怎么跑）、报告 PDF。**不要交**权重 npz 和参考数据。

报错排查心法（课件 slide 73）：**找到第一个开始不同的阶段**，在那一阶段里打印中间数组的 shape 和前几个值，和 `reference_outputs.npz` 对比。绝大多数 bug 是转置错了轴或 reshape 顺序不对——症状都是"shape 对了但数值不对"。

---

*本文档由 AI 辅助整理（课件与学生包内容均为官方材料；TODO 部分只给提示不代做，按课程要求请注明辅助来源）。*
