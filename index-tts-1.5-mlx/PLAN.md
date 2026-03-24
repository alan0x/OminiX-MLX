# IndexTTS 1.5 MLX 实施计划

**目标**: 在 OminiX-MLX 中实现 `index-tts-1.5-mlx` crate，使用 mlx-community/IndexTTS-1.5 权重。
**远期目标**: 通路打通后评估升级到 IndexTTS 2.0。

---

## 进度总览

| Phase | 状态 | 说明 |
|-------|------|------|
| Phase 0：前置调研 | ⚠️ 跳过 | 采用路线 B（从 mlx-audio Python 源码推断 key 命名），在 Phase 3 编译时验证 |
| Phase 1：骨架搭建 | ✅ 完成 | Cargo.toml、error.rs、config.rs、lib.rs 骨架、examples/synthesize.rs |
| Phase 2：子模块实现 | ✅ 完成 | mel.rs、conditioner.rs、gpt.rs、bigvgan.rs、sampling.rs、lib.rs 全线打通 |
| Phase 3：Mac 上验证 | ⏳ 待执行 | 需要在 Mac 上编译、下载权重、端到端测试 |
| Phase 4：质量与性能 | ⬜ 待 Phase 3 后 | MOS 对比、RTF 测量 |
| Phase 5：Moxin-Voice 集成 | ⬜ 远期 | Dora 节点、UI 后端切换 |
| Phase 6：IndexTTS 2.0 评估 | ⬜ 远期 | Phase 3 完成后再决策 |

---

## Phase 0：前置调研

**状态**: ⚠️ 跳过，改为在 Phase 3 运行时验证

**原决策**: 下载权重 → 枚举 key → 建立映射表
**实际采用路线 B**: 直接阅读 `mlx-audio` Python 源码推断 key 命名，置信度约 85%。
主要风险：ECAPA-TDNN（BigVGAN 内部 speaker encoder）的 key 命名使用 speechbrain 约定，Phase 3 可能需要修正。

---

## Phase 1：骨架（✅ 已完成）

- [x] 创建 `index-tts-1.5-mlx/` crate 目录结构
- [x] `Cargo.toml`（package name: `index-tts-15-mlx`，含 tokenizers、hound、mlx-rs 等依赖）
- [x] `src/error.rs`（Error enum：Mlx, Io, Json, MlxIo, WeightLoad, WeightNotFound, Model, Config, Audio, Tokenizer）
- [x] `src/config.rs`（完整反序列化 config.json）
- [x] `src/lib.rs`（IndexTts 公开接口骨架）
- [x] `examples/synthesize.rs`（CLI：--model / --reference / --text / --output）
- [x] 加入根 workspace `Cargo.toml`
- [x] `RESEARCH.md`、`PLAN.md`

---

## Phase 2：子模块实现（✅ 已完成）

所有源文件已实现并推送到 `alan0x/OminiX-MLX` main 分支（commit `81a74a0`）。

### Phase 2.1：mel.rs ✅

**接口**: `mel_spectrogram(&[f32]) -> Result<Array>` → `[1, 100, T]`

实现要点：
- 中心填充（N_FFT/2 两侧）
- 手动分帧 + Hann 窗（CPU）
- `mlx_rs::fft::rfft` 计算功率谱
- HTK Mel filterbank（mel_filterbank + hz_to_mel + mel_to_hz）
- log 压缩，clamp 避免 log(0)
- 公开辅助函数 `array_to_vec(arr) -> Result<Vec<f32>>`

关键修正：
- `ops::abs_squared` 不存在 → `let mag = ops::abs(&x)?; let power = (&mag * &mag)?`
- `mlx_rs::ops::fft::rfft` 路径错误 → `mlx_rs::fft::rfft`
- `Array::from_float(v)` 不存在 → `array!(v)`

### Phase 2.2：conditioner.rs ✅

**接口**:
- `ConformerEncoder::forward(mel: [1,n_mels,T]) -> [1,T',d_model]`
- `PerceiverResampler::forward(context: [1,T',d_enc]) -> [1,N_lat,d_model]`

实现要点：
- FeedForward（SiLU 激活，0.5 残差权重）
- ConformerBlock（FF-macaron + MHSA + GLU 卷积模块 + FF + final norm）
- ConformerEncoder：Conv2dSubsampling（2× stride-2 conv2d，channels-last `[B,T,n_mels,1]`）+ 6 blocks
- PerceiverLayer：latent 对 context 的 cross-attention + FF
- PerceiverResampler：project context + 广播 learnable latents + 2 layers + RMSNorm
- 权重辅助函数：`get_weight`、`load_linear`、`load_layer_norm`、`load_rms_norm`

关键修正：
- MLX conv2d 输入必须是 channels-last `[B,T,n_mels,1]`，而非 `[B,1,T,n_mels]`
- `conv2d` 参数是 `(i32,i32)` 元组，不是 `&[i32]` slice

### Phase 2.3：gpt.rs ✅

**接口**: `Gpt::generate(text_ids, conditioning, max_mel, top_k, top_p, temp) -> (Vec<u32>, Array[1,T,d_model])`

实现要点：
- TransformerBlock：pre-LN + combined QKV c_attn + causal mask（prefill t>1）+ GELU MLP
- 24 层 GPT-2 style decoder
- `build_prefill_embeds`：layout = [conditioning | text_emb+pos | mel_BOS]
- `forward_step` 返回 `(logits [B,vocab], hidden_last [B,d_model])`
- `generate` 累积 GPT hidden states → latent `[1,T,d_model]`（非 input embeddings）

关键修正：
- `ops::full` 需要类型参数和 `array!()` 值：`ops::full::<f32>(&shape, array!(f32::NEG_INFINITY))`
- `Array::zeros` 需要类型参数：`Array::zeros::<f32>(&shape)`
- `ops::concatenate` 需要 `&[&Array]`：先 collect refs

### Phase 2.4：bigvgan.rs ✅

**接口**: `BigVgan::forward(gpt_latent: [1,T,d_model], ref_mel: [1,T_ref,n_mels]) -> [1,T_audio,1]`

实现要点：
- `snake_beta(x, alpha, beta)`: `x + sin²(β·x) / β`，per-channel 参数广播 `[1,1,C]`
- `BatchNorm1d`：推理模式（running stats），输入 `[B,L,C]`
- `SeModule`：global avg pool → FC+ReLU+FC+Sigmoid → channel scaling
- `Bottle2neck`：SE-Res2Net block（scale=8，分层膨胀卷积）
- `EcapaTdnn`：conv1+bn + 3×SE-Res2Net + aggregation + attentive stats pooling（weighted mean+std）+ BN5+FC6+BN6
- `ResBlock`/`ResBlockLayer`：(SnakeBeta+dilated_conv)×n + 残差
- `UpStage`：ConvTranspose1d + trim + FiLM + MRF
- `BigVgan::forward`：spk_emb → conv_pre → global FiLM → N stages → SnakeBeta → conv_post → tanh

关键修正：
- `x.mean(&[1], true)` → `x.mean_axis(1, true)`
- `conv_transpose1d` 需要 `output_padding` 参数：`conv_transpose1d(x, w, stride, 0, 1, 0, 1)`

### Phase 2.5：sampling.rs ✅

**接口**: `sample(logits: &Array, temperature: f32, top_k: usize, top_p: f32) -> Result<u32>`

实现要点（参考 qwen3-tts-mlx/src/sampling.rs 模式）：
- 贪心：`ops::indexing::argmax_axis(logits, -1, None)` + `token.item::<u32>()`
- Temperature scaling
- Top-k：`ops::indexing::topk` → threshold → `ops::r#where`
- Top-p：负数排序 trick + `ops::cumsum` → threshold → `ops::r#where`
- 采样：`mlx_rs::random::categorical`

关键修正：
- 整个文件从初版完全重写，因 MLX API 与预期不符（`ops::greater_equal` 等不存在）

### Phase 2.6：lib.rs 更新 ✅

- `IndexTts` 结构体持有：config、tokenizer、conformer、perceiver、gpt、bigvgan
- `synthesize()` 完整流水线：tokenize → ref_mel → conformer → perceiver → gpt.generate → bigvgan.forward → Vec<f32>
- `load_all_weights()`：支持单文件 `model.safetensors` 和分片 `model.safetensors.index.json`
- `load_tokenizer()`：HuggingFace tokenizers crate 加载 `tokenizer.json`
- `load_wav()` / `save_wav()`：WAV I/O（hound crate）

---

## Phase 3：Mac 上验证（⏳ 待执行）

**前置条件**: 需要 macOS + Apple Silicon（MLX 只支持 macOS）

### 步骤 1：编译

```bash
cd ~/path/to/OminiX-MLX
cargo build -p index-tts-15-mlx
```

预期：首次编译可能报错（weight key 命名不符）。按错误信息修正 `conditioner.rs`、`gpt.rs`、`bigvgan.rs` 中的 key 字符串。

### 步骤 2：下载模型权重

```bash
huggingface-cli download mlx-community/IndexTTS-1.5 \
    --local-dir ~/.OminiX/models/IndexTTS-1.5
```

### 步骤 3：枚举实际权重 key（关键！）

```python
import mlx.core as mx, os

model_dir = os.path.expanduser("~/.OminiX/models/IndexTTS-1.5")

# 单文件
import safetensors
with safetensors.safe_open(f"{model_dir}/model.safetensors", framework="pt") as f:
    for k in sorted(f.keys()):
        print(f"{k:80s} {f.get_tensor(k).shape}")
```

或用 mlx：
```python
import mlx.core as mx
w = mx.load(f"{model_dir}/model.safetensors")
for k, v in sorted(w.items()):
    print(f"{k:80s} {list(v.shape)}")
```

将输出填入本文档 **Appendix** 部分。

### 步骤 4：对比 key 映射，修正代码

重点检查（按置信度从低到高排序）：

| 子模块 | 预期前缀 | 置信度 | 可能的问题 |
|--------|----------|--------|-----------|
| ECAPA-TDNN | `bigvgan.speaker_encoder.*` | 60% | speechbrain key 命名未知 |
| PerceiverResampler | `perceiver_encoder.*` | 80% | layer 编号可能不同 |
| ConformerEncoder | `conditioning_encoder.*` | 85% | block 编号应 OK |
| GPT blocks | `gpt.h.{i}.*` | 90% | 标准 GPT-2 命名 |
| Embeddings | `text_embedding.*`, `mel_embedding.*` | 95% | 顶层，几乎确定 |

### 步骤 5：端到端测试

```bash
cargo run --example synthesize -- \
    --model ~/.OminiX/models/IndexTTS-1.5 \
    --reference path/to/reference.wav \
    --text "你好，世界！" \
    --output output.wav
```

### 步骤 6：验证输出

- 用 macOS 自带播放器或 `afplay output.wav` 播放
- 对比 Python mlx-audio 版本输出：
  ```python
  from mlx_audio.tts.models.indextts import IndexTTS
  model = IndexTTS(model_path="~/.OminiX/models/IndexTTS-1.5")
  audio = model.generate("你好，世界！", reference="reference.wav")
  ```

### 常见错误及修复

**WeightNotFound("bigvgan.speaker_encoder.xxx")**
→ 用 Step 3 的脚本找到实际 key，修改 `bigvgan.rs` 中对应的 `get_weight(weights, "...")` 调用

**Shape mismatch**
→ 打印中间 tensor shape，与 mlx-audio Python 版本对比

**empty audio output**
→ 检查 `gpt.generate` 是否过早返回 stop_mel_token；检查 `bigvgan.forward` FiLM 注入是否正确

---

## Phase 4：质量与性能（⬜ 待 Phase 3 通过后）

- [ ] 与 Python mlx-audio 输出做主观 MOS 对比
- [ ] 测量 RTF（Real-Time Factor），目标 > 1x on M4
- [ ] 内存占用测量，确认 < 2GB
- [ ] 补全 README.md（模型下载、使用示例、性能数据）
- [ ] 修正 mel 归一化（如有必要，对比 Python `log_mel_spectrogram`）

---

## Phase 5：Moxin-Voice 集成（⬜ 远期）

- [ ] 在 `node-hub/` 新增 `dora-index-tts-1.5-mlx` Dora 节点
- [ ] 更新 `apps/moxin-voice/dataflow/tts.yml`
- [ ] UI 侧新增 IndexTTS 后端切换

---

## Phase 6：IndexTTS 2.0 升级评估（⬜ 远期）

完成 Phase 3 后再做决定，评估内容：
- [ ] IndexTTS 2.0 权重 PyTorch → safetensors 转换
- [ ] S2Mel CFM（Flow Matching）实现复杂度评估
- [ ] SeamlessM4T 是否可复用 OminiX-MLX 已有组件
- [ ] 是否在 1.5 基础上增量实现，或新建 `index-tts-2.0-mlx` crate

---

## Appendix：权重 Key 映射表

> 待 Phase 3 Step 3 执行后填入实际 key

以下为基于 mlx-audio 源码推断的**预期** key 前缀：

```
# 子模块                          预期 key 前缀
# ─────────────────────────────────────────────────────────────
# Text embedding                  text_embedding.weight
# Mel code embedding              mel_embedding.weight
# Mel positional embedding        mel_pos_embedding.weight
# Text positional embedding       text_pos_embedding.weight
# GPT transformer blocks          gpt.h.{0..23}.*
#   attention QKV                   gpt.h.{i}.attn.c_attn.{weight,bias}
#   attention output proj           gpt.h.{i}.attn.c_proj.{weight,bias}
#   MLP fc                          gpt.h.{i}.mlp.c_fc.{weight,bias}
#   MLP proj                        gpt.h.{i}.mlp.c_proj.{weight,bias}
#   LayerNorm 1                     gpt.h.{i}.ln_1.{weight,bias}
#   LayerNorm 2                     gpt.h.{i}.ln_2.{weight,bias}
# GPT final norm                  gpt.ln_f.{weight,bias}
# Mel output head                 mel_head.{weight,bias}
# Final norm before mel head      final_norm.{weight,bias}
# ─────────────────────────────────────────────────────────────
# Conformer encoder               conditioning_encoder.*
#   subsampling conv1               conditioning_encoder.embed.conv.0.weight
#   subsampling conv2               conditioning_encoder.embed.conv.2.weight
#   subsampling linear              conditioning_encoder.embed.out.0.{weight,bias}
#   conformer blocks                conditioning_encoder.encoders.{0..5}.*
# ─────────────────────────────────────────────────────────────
# Perceiver resampler             perceiver_encoder.*
#   context projection              perceiver_encoder.proj_context.{weight,bias}
#   learnable latents               perceiver_encoder.latents
#   perceiver layers                perceiver_encoder.layers.{0,1}.*
#   final norm                      perceiver_encoder.norm.weight
# ─────────────────────────────────────────────────────────────
# BigVGAN                         bigvgan.*
#   conv_pre                        bigvgan.conv_pre.weight
#   upsample stages                 bigvgan.ups.{0..N}.*
#   ResBlocks                       bigvgan.resblocks.{0..M}.*
#   conv_post                       bigvgan.conv_post.weight
#   FiLM scale/shift                bigvgan.cond_layer.{weight,bias}
# ECAPA-TDNN (inside BigVGAN)     bigvgan.speaker_encoder.*   ← 最不确定
#   conv1                           bigvgan.speaker_encoder.conv1.{weight,bias}
#   SE-Res2Net layers               bigvgan.speaker_encoder.layer{1,2,3}.*
#   attention                       bigvgan.speaker_encoder.attention.*
#   FC6 / BN6                       bigvgan.speaker_encoder.fc6.*, bn6.*
```
