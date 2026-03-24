# IndexTTS 1.5 MLX 实施计划

**目标**: 在 OminiX-MLX 中实现 `index-tts-1.5-mlx` crate，使用 mlx-community/IndexTTS-1.5 权重。
**远期目标**: 通路打通后评估升级到 IndexTTS 2.0。

---

## Phase 0：前置调研（不写业务代码）

**目标**: 确认权重 key 命名，不踩坑。

### 任务

- [ ] 下载 mlx-community/IndexTTS-1.5 权重到本地
  ```bash
  huggingface-cli download mlx-community/IndexTTS-1.5 \
      --local-dir ~/.OminiX/models/IndexTTS-1.5
  ```

- [ ] 枚举 safetensors key，建立子模块映射表
  ```python
  import mlx.core as mx
  w = mx.load("model.safetensors")
  for k, v in sorted(w.items()):
      print(f"{k:80s} {v.shape}")
  ```

- [ ] 整理 key 前缀 → 子模块对照表（写入本文档 Appendix）

- [ ] 阅读 OminiX-MLX 中 `qwen3-asr-mlx` 的 Conformer 实现，评估可复用程度

**完成标志**: 本文档 Appendix 中填入完整的 key 映射表。

---

## Phase 1：骨架（已完成）

- [x] 创建 `index-tts-1.5-mlx/` crate 目录结构
- [x] `Cargo.toml`
- [x] `src/error.rs`
- [x] `src/config.rs`（config.json 反序列化）
- [x] `src/lib.rs`（公开接口骨架，synthesize 返回 not implemented）
- [x] `examples/synthesize.rs`（CLI 骨架）
- [x] 加入 workspace `Cargo.toml`
- [x] `RESEARCH.md`、`PLAN.md`

---

## Phase 2：子模块实现

### Phase 2.1：Mel 特征提取（`src/mel.rs`）

**输入**: `&[f32]`（24kHz PCM）
**输出**: `mx::Array` shape `[1, 100, T]`

参数（来自 config）：
- sample_rate: 24000, n_fft: 1024, hop: 256, win: 1024, n_mels: 100

检查 `mlx-rs-core` 是否已有 STFT/mel 实现，若有则直接调用，否则新实现。

- [ ] 实现 STFT（或复用 core）
- [ ] 实现 mel filterbank
- [ ] 单元测试：与 Python `torchaudio.transforms.MelSpectrogram` 输出对比

### Phase 2.2：VQVAE Encoder（`src/vqvae.rs`）

**输入**: mel `[1, 100, T]`
**输出**: 离散 codes `[T']`

结构：
- Encoder: 2× Conv1d（stride 下采样）+ 3× ResNet block（kernel=3）
- Codebook: 8192 × 512，最近邻查找

- [ ] 实现 ResNet block（Conv1d + ReLU + skip）
- [ ] 实现 VQ codebook 查找（argmin 欧氏距离）
- [ ] 加载权重，验证 forward 输出 shape
- [ ] 对比测试：Python IndexTTS VQVAE encoder 对同一音频的输出

### Phase 2.3：Conformer Perceiver Conditioner（`src/conditioner.rs`）

**输入**: mel `[1, 100, T]`
**输出**: 条件向量 `[1, N_perceiver, 512]`

结构：
- Conv2d 下采样输入层（input_layer: "conv2d2"）
- 6× Conformer block（dim=?, heads=8, ffn=2048）
- Perceiver 压缩（mult=2，减少 token 数量）

参考 `qwen3-asr-mlx` 的 Conformer 实现，评估是否可直接复用。

- [ ] 确认 Conformer 维度（从 key 映射表推断）
- [ ] 实现 Conv2d 输入层
- [ ] 复用或实现 Conformer blocks
- [ ] 实现 Perceiver 压缩层
- [ ] 验证输出 shape

### Phase 2.4：GPT（`src/gpt.rs`）

**输入**: text_tokens + 条件向量
**输出**: mel codes `[T_gen]` + latent `[1280, T_gen]`

结构：
- Text embedding：12000 vocab → 1280 dim
- Mel code embedding：8194 vocab → 1280 dim
- 条件向量注入方式（cross-attn prefix 或 concatenation，待 Phase 0 确认）
- 24× Transformer decoder block（dim=1280, heads=20）
- 自回归采样（top-k + top-p + temperature）

复用 `mlx-rs-core`：KV cache、SDPA。

- [ ] 确认条件注入方式（来自 Phase 0 key 映射）
- [ ] 实现 GPT block（attention + FFN + 残差 + LayerNorm）
- [ ] 实现自回归采样循环（含 BOS/EOS 处理）
- [ ] 对比测试：与 Python GPT 输出一致

### Phase 2.5：BigVGAN（`src/bigvgan.rs`）

**输入**: GPT latent `[1280, T]` + speaker embedding `[512]`
**输出**: 音频波形 `[T_audio]`，24kHz

结构：
- 输入投影层
- 6× 上采样阶段（rates=[4,4,4,4,2,2]，kernel=[8,8,4,4,4,4]）
- 每阶段：TransposedConv + 多膨胀 ResBlock（kernels=[3,7,11]，dilations=[1,3,5]）
- **SnakeBeta 激活**：`x + (1/β) * sin²(β * x)`（需新实现，OminiX 内无现成）
- Speaker embedding 在每个上采样层注入（Feature-wise Linear Modulation 或 Add）
- 最终 tanh 输出

- [ ] 实现 SnakeBeta 激活函数
- [ ] 实现 MRF（Multi-Receptive Field）残差块
- [ ] 实现 6 级上采样流水线
- [ ] 实现 speaker embedding 注入机制
- [ ] 验证输出：与 Python BigVGAN 输出 MOS 对比

---

## Phase 3：推理流程集成（`src/infer.rs`）

- [ ] 完整 `IndexTts::synthesize()` 实现
  1. 参考音频 → mel → VQVAE codes → Conformer Perceiver → 条件向量
  2. 文本 → BPE tokens
  3. GPT 自回归生成 mel codes + latent
  4. latent + speaker emb → BigVGAN → 波形
- [ ] 实现 BPE tokenizer 加载（sentencepiece `tokenizer.model`）
- [ ] 长文本分句处理（按标点分段，逐段生成）
- [ ] example `synthesize.rs` 跑通端到端

**完成标志**: `cargo run --release -p index-tts-1.5-mlx --example synthesize` 输出可听的语音。

---

## Phase 4：质量与性能

- [ ] 与 Python 原版输出做 MOS 主观对比
- [ ] 测量 RTF（Real-Time Factor），目标 > 1x on M4
- [ ] 内存占用测量，确认 < 2GB
- [ ] README.md 编写（模型下载、使用说明、性能数据）

---

## Phase 5：Moxin-Voice 集成（远期）

- [ ] 在 `node-hub/` 新增 `dora-index-tts-1.5-mlx` Dora 节点
- [ ] 更新 `apps/moxin-voice/dataflow/tts.yml`
- [ ] UI 侧新增 IndexTTS 后端切换

---

## Phase 6：IndexTTS 2.0 升级评估

完成 Phase 3 后再做决定，评估内容：
- [ ] IndexTTS 2.0 权重 PyTorch → safetensors 转换
- [ ] S2Mel CFM（Flow Matching）实现复杂度评估
- [ ] SeamlessM4T 是否可复用 OminiX-MLX 已有组件
- [ ] 是否在 1.5 基础上增量实现，或新建 `index-tts-2.0-mlx` crate

---

## Appendix：权重 Key 映射表

> 待 Phase 0 完成后填入

```
# 示例格式（实际内容待确认）:
# key                                    shape          子模块
# gpt.text_embedding.weight             [12000, 1280]  GPT.text_emb
# gpt.mel_embedding.weight              [8194, 1280]   GPT.mel_emb
# gpt.layers.0.attn.q_proj.weight       [1280, 1280]   GPT.layer[0]
# dvae.encoder.0.weight                 [512, 100, 3]  VQVAE.encoder
# dvae.codebook.weight                  [8192, 512]    VQVAE.codebook
# bigvgan.ups.0.weight                  [...]          BigVGAN.ups[0]
```
