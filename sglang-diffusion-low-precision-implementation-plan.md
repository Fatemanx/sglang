# SGLang-Diffusion Low-Precision Implementation Plan

## 背景与重新定义

- `issue #23035` 是一份 `SGLang-Diffusion (26 Q2)` roadmap，其中 `low precision (nvfp4, mxfp4, fp8)` 只是性能路线上的一个主题，不是已经被拆分好的单点功能需求。
- 当前 `sglang-diffusion` 已经具备一条可运行的低精度主线，至少覆盖：
  - `fp8`
  - `modelopt-fp8`
  - `modelopt-nvfp4`
  - `nunchaku-svdq`
  - `msmodelslim`
- 当前仓库里“存在 repo-wide `mxfp4` 代码”并不等价于“diffusion 已支持 `mxfp4`”。现有 `mxfp4` 主要位于 `srt` 路径，且偏 autoregressive / MoE 场景；diffusion runtime 目前并没有一个现成的、面向通用 DiT dense linear 的 `mxfp4` 路径。
- 因此，这个任务不应被理解为“把已有 `mxfp4` 开关接进 diffusion”，而应被理解为：
  - 先补强当前 diffusion 已支持的低精度主线；
  - 再单独验证 diffusion DiT dense-linear `mxfp4` backend 是否可行；
  - 只有在可行性被证明后，才进入正式主线化。

## 当前代码事实

- diffusion quant registry 当前只注册了 `fp8 / modelopt / modelopt_fp8 / modelopt_fp4 / modelslim`，没有 `mxfp4`。
- diffusion 当前的已验证主线主要围绕 `ModelOpt FP8` 与 `ModelOpt NVFP4` 展开，已经包含：
  - quant config 解析
  - mixed safetensors 过滤
  - precision variant 去重
  - NVFP4 metadata 与 `group_size` 推断
  - offload / backend fallback guardrail
- diffusion 侧当前已有 6 个 `ModelOpt` baseline case，可作为现阶段低精度回归基线：
  - `FLUX.1 modelopt-fp8`
  - `FLUX.2 modelopt-fp8`
  - `Wan2.2 modelopt-fp8`
  - `FLUX.1 modelopt-nvfp4`
  - `FLUX.2 modelopt-nvfp4`
  - `Wan2.2 modelopt-nvfp4`
- 仓库中已经存在可复用的量化验证基础设施，不应重复造轮子：
  - diffusion kernel correctness test
  - end-to-end diffusion CI case
  - trajectory latent similarity 对比工具
  - consistency / CLIP similarity 测试框架

## 目标

- 短期目标：保护并收紧当前 diffusion `fp8 / modelopt-fp8 / modelopt-nvfp4` 主线。
- 中期目标：回答一个更核心的问题: diffusion DiT dense linear 是否存在一条可验证、可维护的 `mxfp4` backend。
- 长期目标：如果上述问题答案为“是”，则以严格收敛的范围交付 diffusion `mxfp4` MVP。

## 非目标

首版工作明确不覆盖以下范围：

- 一次性支持所有 diffusion 模型
- `Wan2.2` 双 transformer 全量 `mxfp4`
- `LTX`
- `AMD / Aiter`
- `diffusers backend`
- 在线量化
- 自动量化工作流
- 在没有 correctness 与 e2e 证据之前，把 `mxfp4` 写进正式用户支持矩阵

## 总体策略

整个工作拆为三个阶段：

- `P0`: 审计并补强当前已支持低精度主线
- `P1`: 做一个严格收敛的 diffusion `mxfp4` feasibility spike
- `P2`: 仅在 spike 证明可行后，交付正式 diffusion `mxfp4` MVP

核心原则：

- 不先承诺 `mxfp4` 一定能进入主线。
- 不把 loader/config 接口补出来就当作“支持已完成”。
- 不在没有 checkpoint contract 与 backend proof 的情况下扩大战线。

## P0: 审计并补强现有 diffusion 低精度主线

### 目标

把当前已支持的 diffusion 低精度路径做成一个稳定、边界清晰、可回归的基线，为后续 `mxfp4` spike 提供可信地基。

### 范围

仅覆盖以下已支持 family：

- `fp8`
- `modelopt-fp8`
- `modelopt-nvfp4`

### 必查点

- `transformer-path` 与 `transformer-weights-path` 的优先级与覆盖语义是否一致。
- mixed safetensors 过滤逻辑是否稳定。
- precision variant 去重逻辑是否稳定。
- NVFP4 metadata / `group_size` 推断是否稳定。
- `dit_cpu_offload`、`dit_layerwise_offload`、Blackwell backend fallback 的 guardrail 是否仍成立。
- 文档、代码、测试是否表达同一套真实支持范围。
- `modelopt` / `modelopt_fp8` 这类历史 alias 是否会增加后续维护成本；至少需要在计划中显式记录其行为边界。

### 交付物

- 针对现有低精度路径的测试补强。
- 如有必要，对文档和错误信息做校准。
- 一份简短 audit 结论，记录当前主线的真实能力边界与遗留技术债。

### 退出标准

- 现有 6 个 diffusion `ModelOpt` case 持续可作为必过回归基线。
- 与 quant config 解析、mixed export、metadata 推断、offload/backend fallback 相关的单测不回退。
- 文档、测试与代码对“当前支持范围”的描述保持一致。

### 注意事项

- `P0` 是短平快的 preflight audit，不应膨胀成大规模重构。
- `P0` 的目标是收紧现有主线，而不是在此阶段顺手引入 `mxfp4` 用户接口。

## P1: diffusion `mxfp4` feasibility spike

### 目标

回答一个单一问题：

> diffusion 的 DiT dense linear，是否存在一条可验证、可维护、值得继续推进的 `mxfp4` backend。

这一步的目标不是“直接合并一个用户可用功能”，而是“验证路线是否成立”。

### 范围收敛

- 硬件限定为 `Blackwell CUDA`
- 模型限定为 `FLUX` 单 DiT 路径
- 首选 `FLUX.1` 作为验证对象，如果 checkpoint/export 条件不满足，再退而使用 `FLUX.2`
- 仅支持 pre-quantized transformer override
- 不扩展到双 transformer、视频流水线、多平台、多 backend

### 为什么优先不从 FLUX.2 开始

- `FLUX.2` 现有 NVFP4 路径已经包含 packed-QKV 相关特殊处理。
- 如果一开始就以 `FLUX.2` 为默认验证对象，容易把以下问题耦合在一起：
  - checkpoint layout 问题
  - packed QKV loader 问题
  - dense-linear `mxfp4` backend 可行性问题
- 因此，spike 应尽量优先选择 layout 更简单、模型行为更可控的单 DiT 验证对象。

### P1 之前必须先定的 checkpoint contract

在写 kernel / backend 代码之前，必须先冻结一版最小 checkpoint contract：

- `P1 / P2` 阶段的 internal family name 是什么
  - 不建议一开始就把首版内部实现直接命名为对外通用的 `mxfp4`
  - 建议在 `P1 / P2` 期间使用更具体的内部名字，例如 `mxfp4_dit_blackwell`
  - 等支持范围从 `Blackwell + single-DiT + pre-quantized override` 扩大后，再决定是否暴露统一对外 alias `mxfp4`
- 首版工件格式是什么
  - 建议优先选择“带 `config.json` 的 converted transformer component directory”
  - 优先走 `--transformer-path`
  - `--transformer-weights-path` 留给后续 raw export 兼容
- checkpoint 中是否允许 BF16 fallback 层
- fallback 层如何记录
  - config ignore list
  - metadata
  - 或 safetensors tensor-family 推断
- 首版是否允许 packed QKV
  - 如果 packed QKV 不是 spike 必需条件，则明确延后

如果以上 contract 不能先收敛，后续 loader 与测试都会持续摇摆。

### P1 建议采用的最小 on-disk schema

为了避免 loader / auto-detection 在实现中反复摇摆，建议 `P1` 先收敛到一份最小 schema：

```json
{
  "quantization_config": {
    "quant_method": "mxfp4_dit_blackwell",
    "quant_type": "MXFP4",
    "quantization": {
      "quant_algo": "MXFP4",
      "exclude_modules": [],
      "group_size": 32,
      "checkpoint_uses_packed_qkv": false
    }
  }
}
```

说明：

- `quant_method` 使用 `P1 / P2` 的内部 family name，而不是提前承诺通用 `mxfp4`
- `quant_type` / `quant_algo` 明确表达数值格式
- `exclude_modules` 用于记录 BF16 fallback 层
- `group_size`、packed-QKV 等 layout 相关字段必须显式落盘，不依赖隐式推断

如果最终实现需要调整 key 名，必须在 `P1` kickoff 时一次性改定，不要在实现中期继续改 schema。

### P1 建议采用的 discovery order 与 fail policy

`P1 / P2` 不应复用过于宽松的“猜格式”思路，而应优先采用保守的 discovery order：

1. 显式 `--transformer-path` 指向的 override `config.json`
2. 显式 `--transformer-weights-path` 中的 safetensors metadata
3. 如果以上都缺失，则直接 hard fail

首版不建议支持以下模糊行为：

- 从 base model `config.json` 倒推 `mxfp4`
- 在 metadata 缺失时自动猜测 `mxfp4`
- 在首版就为 raw export / mixed export / packed-QKV 全部做 silent fallback

这里的原则是：首版 contract 宁可更严格，也不要为了“自动识别更聪明”而引入脆弱路径。

### P1 需要证明的能力

- diffusion quant registry 能识别新 family
- loader 能解析 checkpoint contract
- dense linear quant method 能在 diffusion `Replicated / Column / RowParallelLinear` 上工作
- 至少有一条可复现的权重后处理与 runtime GEMM 路径
- 至少有一组代表性 shape 的 correctness 证据
- 至少有一份 backend viability 说明，解释为什么选定 backend 适合目标 DiT shape，而不是“只是刚好能跑”

### P1 必须先冻结的 representative shape set

`P1` 不应只拿一个随手挑的 shape 证明“能算”。在开始实现前，必须从所选 `FLUX` checkpoint 中冻结一组代表性 dense-linear shape，至少覆盖：

- 一组 attention projection shape
- 一组 attention out projection shape
- 一组 MLP expansion projection shape
- 一组 MLP down projection shape
- 一组 padding / alignment regression shape

要求：

- exact `(M, N, K)` 清单必须在 `P1` 开工时记录进 repo
- synthetic correctness 与 profiling 必须使用同一组代表性 shape
- 如果 `FLUX.1` 与 `FLUX.2` 的代表 shape 差异足够大，必须在 `P1` 结论中明确当前证据适用于哪一个模型族

### P1 交付物

- 一条最小化的 `mxfp4` prototype 路径，范围严格限定在单 DiT dense linear
- 一份 checkpoint contract 说明
- 一份 representative shape 清单
- 一组 synthetic correctness 结果
- 一份 representative shape profiling / backend choice note
- 一份 BF16 vs `mxfp4` 的 trajectory latent similarity 报告
- 一次端到端 smoke 结果

### P1 成功标准

必须同时满足：

- synthetic dense-linear correctness test 通过
- 代表性 shape 的 profiling 能支持所选 backend 决策，而不是只证明“能运行”
- BF16 与候选 `mxfp4` 路径的 trajectory latent similarity 通过已冻结的数值门槛
- 至少一次 FLUX 端到端生成成功
- 实现复杂度没有明显失控，且后续主线化路径清晰

### P1 止损标准

满足以下任一条件时，应停止继续主线化：

- 找不到可维护的 dense-linear backend
- checkpoint contract 过于脆弱，只能依赖大量模型特例
- correctness 无法稳定通过
- 端到端只能通过堆叠过多临时 workaround 才能跑通

如果止损，应输出阻塞结论，而不是提交半成品接口。

## P2: 正式 diffusion `mxfp4` MVP

### 前置条件

只有在 `P1` 明确证明路线可行后，才进入 `P2`。

### MVP 范围

- `Blackwell CUDA`
- `FLUX`
- 单 DiT
- pre-quantized transformer override
- 一个在 `P1` 中已经冻结的 checkpoint contract

### 交付内容

- diffusion quant registry 接入
- quant config class
- quant config 解析与 loader 接入
- weight load / post-load processing
- dense-linear quant method
- synthetic correctness test
- trajectory latent similarity 验证脚本或固定 recipe
- 一条端到端 smoke case
- 在证据充足后，再补正式文档

### 不在 MVP 首发范围内的能力

- 通用多模型覆盖
- raw export 与 converted component 的双格式全覆盖
- 多平台支持
- AMD / ROCm 路径
- diffusers backend `mxfp4`
- 多 transformer full-model `mxfp4`

## 测试与验收

`mxfp4` 进入主线前，验收必须分层，而不是只看“能不能 load”。

### 第 1 层: synthetic dense-linear correctness

必须先有：

- 代表性 GEMM shape correctness test
- 权重打包 / 后处理 correctness test
- 与 dequant ref 或高精度 ref 的数值对比

在这一层没有通过之前，不应继续声明“runtime 支持已完成”。

### 第 2 层: trajectory latent similarity

必须补充：

- BF16 reference vs candidate `mxfp4`
- 固定 prompt / seed / resolution / step count
- 记录每步 latent cosine / error 指标

仓库已有 trajectory 对比工具，应直接复用，而不是另起一套临时指标。

这一层不能只写“可接受”，必须在 `P1` 中冻结数值门槛。推荐流程：

1. 先采集 `BF16 vs BF16` 重复运行噪声基线
2. 再采集现有 diffusion `NVFP4` 路径的参考分布
3. 基于这两组数据，冻结 `MXFP4` 的通过阈值

在阈值冻结之前：

- trajectory similarity 报告是必须交付的 artifact
- 但不应把主观判断写成“已通过”

在阈值冻结之后：

- `P2` 的实现必须以该阈值作为明确 gate
- 阈值应以可复用的测试配置或配置文件形式记录到 repo 中

### 第 3 层: 端到端 smoke / consistency

至少应有：

- 一条固定模型、固定 prompt 的 e2e smoke
- 如有可用 GT，再接 consistency / CLIP similarity 门槛

如果首版暂时只做 smoke，也必须明确写清楚“尚未进入正式 consistency 支持矩阵”。

### 性能与资源报告

因为该任务来自 diffusion performance roadmap，MVP 至少需要一份最小性能证据：

- BF16 vs `mxfp4` 的显存占用对比
- BF16 vs `mxfp4` 的单次延迟或吞吐对比
- representative dense-linear shape 的 kernel time / backend profiling 结果

没有数据时，不应写入任何性能收益承诺。

建议把最小性能 recipe 明确为：

- 固定模型、固定 prompt、固定 seed 的 e2e latency
- 固定配置下的 peak GPU memory
- representative shape set 的 microbenchmark / profiler 数据
- 至少一份 torch profiler 或等价 trace，用于说明 backend 选择不是拍脑袋决定

## PR 策略

建议不要把所有工作塞进一个 PR。

### PR 1: `P0` audit / stabilization

- 只收紧现有 `fp8 / nvfp4`
- 不引入 `mxfp4` 用户接口

### PR 2: `P1` spike

- 可以是 draft PR、实验分支或内部验证分支
- 重点是回答“路线是否成立”

### PR 3: `P2` MVP

- 仅在 `P1` 成功后推进
- 首发范围严格收敛

如果 `P1` 失败，则只保留 `PR 1` 的主线补强成果，并把 `mxfp4` 结论记录为阻塞项。

## 开工前待决策问题

- `P1 / P2` 阶段的 internal family name 是否采用更具体命名，例如 `mxfp4_dit_blackwell`
- 首版 checkpoint contract 是否统一为 converted transformer directory + `--transformer-path`
- `P1` 的首个验证模型是否能使用 `FLUX.1`
- packed QKV 是否明确延后
- representative shape set 是否已经冻结并记录
- trajectory similarity 阈值准备采用什么冻结流程
- `P2` 的首条 e2e case 是先做 smoke，还是直接进入 consistency GT 流程

这些问题应在开始写实现前定掉，否则会持续反复修改 loader 与测试边界。

## CI 与自动化落点

这份计划中的验证不应只停留在“应该做什么”，而应尽早绑定到明确的自动化落点。

### pre-merge 必须绑定的 lane

- synthetic dense-linear correctness
  - 建议接入 `B200` kernel/unit lane，风格与现有 diffusion `NVFP4` correctness test 对齐
- diffusion e2e smoke
  - 建议接入现有 diffusion `B200` test lane，作为首条 `MXFP4` smoke case

### `P1` 阶段必须产出的非 gating artifact

- representative shape profiling 报告
- trajectory latent similarity 报告
- backend choice note

这些 artifact 在 `P1` 阶段可以先不作为 pre-merge blocker，但必须随 spike 结果一起提交并归档。

### `P2` 阶段需要升级为明确 gate 的内容

- trajectory latent similarity 阈值检查
- 至少一条固定模型的 e2e smoke

如果这些检查在 `P2` 阶段仍无法稳定自动化，说明路线尚未成熟，不应宣称 `mxfp4` 已正式支持。

## 关键代码定位

- `docs/diffusion/quantization.md`
  - diffusion 低精度能力的现行说明入口；当前 6 个 `ModelOpt` 基线 case 也在这里有明确矩阵。
- `python/sglang/multimodal_gen/runtime/layers/quantization/__init__.py`
  - diffusion quant registry；可直接看出当前尚未接入 `mxfp4`。
- `python/sglang/multimodal_gen/runtime/utils/quantization_utils.py`
  - quant config 解析、ModelOpt family 归一化、NVFP4 metadata 推断与 safetensors 聚合逻辑。
- `python/sglang/multimodal_gen/runtime/loader/transformer_load_utils.py`
  - transformer override 的 quant load spec、mixed safetensors 选择、precision variant 去重、ModelOpt NVFP4 merge 逻辑。
- `python/sglang/multimodal_gen/runtime/layers/quantization/modelopt_quant.py`
  - diffusion `modelopt-fp8` 与 `modelopt-nvfp4` 的核心实现，可作为 `mxfp4` dense-linear 接入时的最近参考。
- `python/sglang/multimodal_gen/runtime/layers/quantization/modelopt_fp8.py`
  - 另一条历史 `ModelOpt FP8` config 实现；后续若扩展 quant family，需注意 alias / 维护成本问题。
- `python/sglang/multimodal_gen/runtime/layers/linear.py`
  - diffusion 各类 linear layer 的 quant method 接入点。
- `python/sglang/multimodal_gen/runtime/models/dits/flux_2.py`
  - 现有 FLUX.2 NVFP4 packed-QKV 特殊处理入口；如果 `P1` 选择 FLUX.2，需要特别审视这里。
- `python/sglang/srt/layers/quantization/mxfp4.py`
  - repo-wide `mxfp4` 现有实现入口，主要用于 `srt`；能提供背景，但不能直接等同于 diffusion dense-linear 支持。
- `python/sglang/multimodal_gen/test/server/gpu_cases.py`
  - diffusion 端到端低精度 test case 入口；当前 6 个 `ModelOpt` baseline 就在这里。
- `python/sglang/multimodal_gen/test/unit/test_transformer_quant.py`
  - mixed export、precision variant、NVFP4 config 解析、offload guardrail 等现有单测入口。
- `python/sglang/jit_kernel/tests/diffusion/test_diffusion_nvfp4_scaled_mm.py`
  - diffusion NVFP4 correctness 测试，可作为 future `mxfp4` kernel-level 验证的参照。
- `python/sglang/multimodal_gen/tools/compare_diffusion_trajectory_similarity.py`
  - BF16 与量化候选路径做 trajectory latent similarity 对比的现成工具。

## 最终结论

- 当前最合理的推进方式不是“直接实现 diffusion `mxfp4` 支持”，而是：
  - 先把现有 diffusion `fp8 / nvfp4` 主线收紧；
  - 再用一个严格收敛的 `Blackwell + FLUX + single-DiT + pre-quantized override` spike 证明 dense-linear `mxfp4` backend 可行；
  - 只有在这条路线被证明成立后，才交付正式 diffusion `mxfp4` MVP。
- 如果最终证明路线不成立，正确结果是：
  - 合入现有主线的补强；
  - 记录 `mxfp4` 的阻塞结论；
  - 不提交半成品支持，也不暴露误导性的用户接口。
