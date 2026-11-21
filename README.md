# Project1 作业说明
## 六个 TODO 完成的情况

| TODO | 位置 | 一句话做法 |
|---|---|---|
| 01 patchify | `ImageEncoder.patchify` | `[B,3,224,224]`→拆 7×7 网格→`transpose(0,2,4,1,3,5)`→`[B,49,3072]` |
| 02 split_heads | `split_heads` | `[B,N,768]`→`reshape(B,N,12,64)`→`transpose(0,2,1,3)`→`[B,12,N,64]` |
| 03a 分数 | `__call__` | `q @ k.swapaxes(-1,-2) / sqrt(64)` → `[B,H,N,N]` |
| 03b 混合 | `__call__` | `coef @ v` → `[B,H,N,64]` → 换轴拼回 `[B,768]` |
| 03c 投影 | `__call__` | `merged @ out_weight.T + out_bias` |
| 04a/b block | `TransformerBlock` | 残差 `x+update`；MLP `GELU(u@W1.T+b1)@W2.T+b2` 再残差 |
| 05 embed | `ImageEncoder.embed` | `patches@W.T`（无偏置）→ 拼 CLS → 加 position |
| 06 输出 | `ImageEncoder.__call__` | 取 `[:,0,:]`→post_norm→`@proj`（不转置）→ 单位化 |

## 运行代码如下

```bash
# 用带 numpy 的解释器；本机 C:/Python314/python.exe 已验证
python check.py --stage all   
python run.py --output output_tokens.npz  
python transformer_steps.py    
```

## 实测结果（C:/Python314，numpy 2.5.1）

- assets / preprocess / patches / tokens / block0 / last_tokens / output / batch ** PASS**。
- output 最大绝对误差 `5.44e-07`（容差 `2e-05`）；batch `4.62e-07`。
- `output_tokens.npz`：shape `(10,512)`、float32、每行模长 1.0。

