# IndexTTS 1.5 前期调研报告

**日期**: 2026-03-24
**目标版本**: IndexTTS 1.5 (mlx-community/IndexTTS-1.5)
**远期版本**: IndexTTS 2.0（通路打通后评估）

---

## 模型来源

- **原始模型**: IndexTeam/IndexTTS-1.5（Bilibili AI 团队）
- **MLX 转换版**: mlx-community/IndexTTS-1.5（已有，可直接使用）
- **参数量**: 0.7B
- **许可证**: Apache 2.0

---

## 推理流程

```
参考音频 .wav (24kHz)
    │
    ▼ MelSpectrogram [1, 100, T_ref]
    │   n_fft=1024, hop=256, win=1024, n_mels=100
    │
    ▼ VQVAE Encoder → 离散 codes [T_ref]
    │   codebook size: 8192, hidden_dim: 512
    │   (编码参考音色的离散表示)
    │
    │              文本
    │               │ BPE tokenizer (sentencepiece, vocab=12000)
    │               ▼ text_tokens [B, L]
    │
    ▼ Conformer Perceiver Conditioner
    │   (从参考 mel + codes 提取音色条件向量 [512])
    │   6 blocks, 8 heads, Conv2d 下采样输入
    │
    ▼ GPT 自回归解码
    │   24层 Transformer, dim=1280, heads=20
    │   输入: text_tokens + 音色条件向量
    │   输出: mel codes [T_gen] + latent [1280, T_gen]
    │
    ▼ BigVGAN 声码器
    │   upsample_rates=[4,4,4,4,2,2], 总倍率=512
    │   SnakeBeta 激活, 多级残差块
    │   条件输入: speaker embedding [512] + GPT latent
    │
    ▼ 音频波形 [T_audio], 24kHz
```

---

## 三个核心子模块

### 1. VQVAE（dvae）

| 参数 | 值 |
|------|-----|
| 输入 | 100-dim mel frames |
| 编码器层数 | 2层卷积 + 3个 ResNet block |
| Codebook 大小 | 8192 tokens |
| Codebook 维度 | 512 |
| kernel_size | 3 |
| 推理只用 | Encoder（Decoder 用于训练） |

**实现复杂度**: 低

### 2. GPT + Conformer Perceiver

| 参数 | 值 |
|------|-----|
| 层数 | 24 |
| 模型维度 | 1280 |
| 注意力头数 | 20 |
| 文本词表 | 12000 |
| Mel 码词表 | 8194（含 BOS/EOS）|
| 条件模块 | Conformer Perceiver |
| Conformer blocks | 6 |
| Perceiver 压缩比 | 2 |
| 输入层类型 | conv2d2（2D 卷积下采样）|

GPT 是标准 decoder-only transformer，KV cache 和 SDPA 可复用 mlx-rs-core 已有实现。
Conformer Perceiver 是音色条件提取器，需单独实现。

**实现复杂度**: 中

### 3. BigVGAN

| 参数 | 值 |
|------|-----|
| 输入 | GPT latent [1280, T] + speaker emb [512] |
| 上采样率 | [4, 4, 4, 4, 2, 2]（总 ×512）|
| 初始通道数 | 1536 |
| 残差块 kernel | [3, 7, 11] |
| 激活函数 | **SnakeBeta**（非标准，需手动实现）|
| 条件注入 | 每个上采样层都注入 speaker embedding |
| 输出采样率 | 24kHz |

BigVGAN 是最复杂的模块，原因：
- SnakeBeta 激活函数 OminiX-MLX 内无现成实现
- 多级转置卷积（6级），含复杂残差结构
- Speaker embedding 在每层注入

**实现复杂度**: 高

---

## 权重文件现状

mlx-community/IndexTTS-1.5 已将三个子模块合并为单一文件：

| 文件 | 大小 |
|------|------|
| `model.safetensors` | 1.43 GB |
| `config.json` | 4.21 kB |
| `tokenizer.model` | 476 kB（sentencepiece BPE）|

**关键待确认事项（Phase 0 任务）**：
`model.safetensors` 内各子模块的 key 前缀命名，例如：
- `gpt.*` / `dvae.*` / `bigvgan.*`？
- 还是其他命名方式？

需要下载权重文件后通过 `mx.load()` 枚举 key 来确认。

---

## 与现有 OminiX-MLX crate 的关系

| 组件 | 可复用 |
|------|--------|
| KV cache | `mlx-rs-core` ✅ |
| SDPA（Scaled Dot-Product Attention）| `mlx-rs-core` ✅ |
| RoPE | `mlx-rs-core` ✅（GPT 若使用 RoPE）|
| Conformer blocks | `qwen3-asr-mlx` 参考实现 |
| SnakeBeta 激活 | **需新增** ❌ |
| VQVAE | **需新增** ❌ |
| BigVGAN 上采样块 | **需新增** ❌ |

---

## IndexTTS 2.0 vs 1.5 差异（远期参考）

| 新增模块 | 作用 | 实现难度 |
|---------|------|---------|
| SeamlessM4T | 语义特征提取（16kHz）| 高（独立大模型）|
| CAMPPlus | 说话人风格提取 | 中 |
| QwenEmotion | 8维情感向量提取 | 中（Qwen fine-tune）|
| S2Mel CFM | Flow Matching 扩散，语义码→mel | **极高** |

2.0 无现成 MLX 权重，需自行转换。S2Mel CFM 是主要难点。

---

## 结论

- **1.5 实现路径清晰，可行**
- **权重已有 MLX 版本，省去转换步骤**
- **最大技术难点：BigVGAN（SnakeBeta + 多级上采样）**
- **2.0 升级条件：1.5 通路验证 + S2Mel CFM 调研**
