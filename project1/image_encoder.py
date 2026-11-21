import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']      # [D]
        self.heads = heads                       # 头数 H=12
        self.width = self.out_bias.shape[0]      # 单头之前的宽度 D=768
        self.head_dim = self.width // heads      # 每头宽度 dh=64

    def split_heads(self, x):
        """TODO-02：[B,N,D] -> [B,H,N,D/H]，把特征轴切成多个头。

        D=768，H=12，dh=64。先 reshape 拆出 H 与 dh，
        再把 H 挪到 N 前面；token 顺序 N 不变。
        """
        B, N, D = x.shape                        # 读入形状
        H, dh = self.heads, self.head_dim        # 12, 64
        x = x.reshape(B, N, H, dh)               # [B,N,H,dh] 连续切分特征
        return x.transpose(0, 2, 1, 3)           # [B,H,N,dh] 头提前

    def __call__(self, x, causal=False):
        # 已给：线性权重存 [out,in]，故乘 .T；参数固定不动。
        qkv = x @ self.qkv_weight.T + self.qkv_bias  # [B,N,3D]
        q = self.split_heads(qkv[:, :, :self.width])           # 前 D 段 -> q
        k = self.split_heads(qkv[:, :, self.width:2*self.width])  # 中段 -> k
        v = self.split_heads(qkv[:, :, 2*self.width:])         # 后 D 段 -> v

        # TODO-03a：算 q·k 并除以 sqrt(dh)；分数形状 [B,H,N,N]。
        # q,k,v 都是 [B,H,N,64]。k 只交换最后两轴，保住 B/H 批组。
        scale = np.sqrt(np.float32(self.head_dim))  # float32 标量，防升成 float64
        scores = (q @ k.swapaxes(-1, -2)) / scale   # [B,H,N,N]

        # 已给：文本塔才用的因果遮罩；图像 causal=False，不触发。
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # softmax -> 每行和为 1

        # TODO-03b：用权重混合 v 的行，再把多头拼回 [B,N,D]。
        context = coefficients @ v                # [B,H,N,64] 加权求和
        B, H, N, dh = context.shape
        # 先把 H 挪回 N 后，再摊平最后两轴成整条特征。
        merged = context.transpose(0, 2, 1, 3).reshape(B, N, H * dh)  # [B,N,768]

        # TODO-03c：输出投影。out_weight 存 [out,in]，故乘 .T。
        out = merged @ self.out_weight.T + self.out_bias  # [B,N,D]
        return out


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']    # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']        # [4D]
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']      # [D]
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # 已给：先归一化每个 token，再算注意力更新。
        update = self.attention(self.norm1(x), causal=causal)  # [B,N,D]

        # TODO-04a：第一条残差。注意加的是原始 x，不是 norm1(x)。
        residual = x + update                    # [B,N,D]

        # 已给：第二次归一化，变量名 u 与配套讲解一致。
        u = self.norm2(residual)                 # [B,N,D]

        # TODO-04b：MLP 两线性层 + GELU，再接第二条残差。
        hidden = u @ self.fc1_weight.T + self.fc1_bias  # [B,N,3072] 升维
        hidden = self.gelu(hidden)                      # 精确 erf 激活，形状不变
        mlp_update = hidden @ self.fc2_weight.T + self.fc2_bias  # [B,N,768] 降维
        return residual + mlp_update             # [B,N,D] 第二条残差


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]         # 12 个结构相同、参数不同的 block
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']    # [768,512]，已是 [in,out]

    def patchify(self, images):
        """TODO-01：[B,3,224,224] -> [B,49,3072]。

        224/32=7，切成 7x7=49 个 patch；每块 3*32*32=3072 个值。
        遍历顺序：先左到右、再上到下；块内按 C,H,W 展平。
        """
        B, C, H, W = images.shape                 # 3,224,224
        p = 32                                   # patch 边长
        gh, gw = H // p, W // p                  # 7,7 网格
        # 拆成 [B,C,格行,p,格列,p]
        x = images.reshape(B, C, gh, p, gw, p)
        # 调轴成 [B,格行,格列,C,p,p]：网格先、通道后、块内最后。
        x = x.transpose(0, 2, 4, 1, 3, 5)
        # 摊平最后三轴 C*p*p=3072，得 [B,49,3072]。
        return x.reshape(B, gh * gw, C * p * p)

    def embed(self, images):
        patches = self.patchify(images)          # [B,49,3072]

        # TODO-05a：把 patch 投影到 D=768。无偏置；权重存 [out,in]，故 .T。
        patch_tokens = patches @ self.patch_weight.T  # [B,49,768]

        # 已给：每张图一行 CLS，形状 [B,1,768]。
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # TODO-05b：token 轴上把 CLS 拼在最前，再加可学习位置编码。
        tokens = np.concatenate([cls, patch_tokens], axis=1)  # [B,50,768]
        tokens = tokens + self.position                       # [50,768] 广播到 B
        return tokens                            # [B,50,768]

    def encode_tokens(self, images):
        """已给：拼好 token 输入并过 12 个 block；[B,50,768]。"""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06：每张图输出一个归一化向量 [B,512]。"""
        tokens = self.encode_tokens(images)      # [B,50,768]

        # 取第 0 行 CLS -> [B,768]；单独的后归一化 -> [B,768]。
        cls_token = tokens[:, 0, :]              # 只选 CLS，丢掉 49 个 patch
        normalized = self.post_norm(cls_token)
        # 输出投影：visual.proj 已是 [in,out]，不要转置。
        output = normalized @ self.projection    # [B,768] @ [768,512] -> [B,512]

        # 已给：最后拉成单位向量。
        return unit_vector(output)
