# SGLang-Diffusion Low-Precision Execution Checklist

本文档是 [sglang-diffusion-low-precision-implementation-plan.md](/data1/lyxu18/opensource_proj/sglang/sglang-diffusion-low-precision-implementation-plan.md) 的执行版 checklist。

目标不是重复设计，而是让后续 session 可以直接按 checklist 推进，并留下可审计的证据。

## 使用方式

- 每次开始新 session 时，先阅读：
  - 本文档
  - `sglang-diffusion-low-precision-implementation-plan.md`
- 每完成一个 checklist 项目，就在本文件中更新：
  - 状态
  - 实际输出物路径
  - 测试 / benchmark 证据
  - 未决问题
- 不允许跳过 `P1` 直接进入 `P2`。
- 除非 `P1` 成功标准全部满足，否则不能宣称 diffusion `mxfp4` 已支持。

## 全局规则

- [ ] 不扩大范围到以下非目标：
  - `Wan2.2` 双 transformer 全量 `mxfp4`
  - `LTX`
  - `AMD / Aiter`
  - `diffusers backend`
  - 在线量化
  - 自动量化工作流
- [ ] 不在缺少 correctness 与 e2e 证据时更新正式支持矩阵。
- [ ] 不把 loader / config 入口补齐视为“功能完成”。
- [ ] 所有关键结论都要有 artifact，而不是只留在聊天记录中。
- [x] 使用conda的exp_env的环境完成
- [x] 主agent只负责任务分发、任务验收；具体任务由sub agent完成实现、自测

## 状态总表

| 阶段 | 状态 | 负责人 / Session | 完成证据 |
| --- | --- | --- | --- |
| `P0` audit / stabilization | `BLOCKED` | 2026-05-26/27 Codex session | `artifacts/diffusion_mxfp4/plan_notes/p0-audit-note-2026-05-26.md`; `artifacts/diffusion_mxfp4/plan_notes/p0-server-baseline-followup-2026-05-27.md` |
| `P1` `mxfp4` feasibility spike | `TODO` |  |  |
| `P1` Go / No-Go decision | `TODO` |  |  |
| `P2` `mxfp4` MVP | `TODO` |  |  |

推荐状态值：

- `TODO`
- `IN_PROGRESS`
- `BLOCKED`
- `DONE`
- `WONT_DO`

## Session 启动 Checklist

每个新 session 开始时先打勾：

- [x] 阅读当前 plan 与 checklist 的最新版本。
- [x] 阅读上一 session 留下的 handoff 记录。（未发现现成 handoff）
- [x] 明确本次 session 只推进一个阶段内的一组任务，不跨阶段乱改。
- [x] 明确本次 session 的输出物会保存到哪里。
- [x] 明确本次 session 是否需要跑测试、benchmark、profile。

## 统一输出物约定

建议统一把中间产物放在以下目录，便于后续 session 复用：

- [x] `artifacts/diffusion_mxfp4/plan_notes/`
- [x] `artifacts/diffusion_mxfp4/profiles/`
- [x] `artifacts/diffusion_mxfp4/trajectory/`
- [x] `artifacts/diffusion_mxfp4/benchmark/`
- [x] `artifacts/diffusion_mxfp4/checkpoint_contract/`

如果实际路径不同，必须在本 checklist 中记录。

---

## P0 Checklist: Audit And Stabilize Existing Diffusion Low Precision

### P0.1 Scope Freeze

- [x] 明确 `P0` 只覆盖：
  - `fp8`
  - `modelopt-fp8`
  - `modelopt-nvfp4`
- [x] 明确 `P0` 不引入任何 `mxfp4` 用户接口。

### P0.2 Read / Audit Existing Paths

- [x] 检查 diffusion quant registry 当前能力。
- [x] 检查 `transformer-path` 与 `transformer-weights-path` 解析与优先级。
- [x] 检查 mixed safetensors 过滤逻辑。
- [x] 检查 precision variant 去重逻辑。
- [x] 检查 NVFP4 metadata / `group_size` 推断逻辑。
- [x] 检查 `dit_cpu_offload` / `dit_layerwise_offload` / Blackwell fallback guardrail。
- [x] 检查 `modelopt` / `modelopt_fp8` 历史 alias 的行为边界。

建议关注代码：

- [x] `python/sglang/multimodal_gen/runtime/layers/quantization/__init__.py`
- [x] `python/sglang/multimodal_gen/runtime/utils/quantization_utils.py`
- [x] `python/sglang/multimodal_gen/runtime/loader/transformer_load_utils.py`
- [x] `python/sglang/multimodal_gen/runtime/layers/quantization/modelopt_quant.py`
- [x] `python/sglang/multimodal_gen/runtime/layers/quantization/modelopt_fp8.py`

### P0.3 Baseline Tests

- [ ] 确认当前 6 个 diffusion `ModelOpt` baseline case 仍是有效回归基线。
- [x] 确认相关 unit test 存在并能覆盖：
  - quant config 解析
  - mixed export
  - precision variant filtering
  - NVFP4 metadata 推断
  - offload / fallback guardrail
- [x] 如存在空白，补充测试。

### P0.4 Docs / Error Messages

- [x] 检查 `docs/diffusion/quantization.md` 是否与真实支持范围一致。
- [x] 如实现与文档不一致，优先收敛描述，不夸大能力。
- [x] 如报错信息不够清晰，补充最小必要错误信息。

### P0.5 P0 Deliverables

- [x] 输出一份简短 audit note。
- [x] 记录发现的技术债与不在 `P0` 修复的事项。
- [x] 提交必要测试 / 文档 / 小修正。

### P0 Exit Gate

只有全部满足，才能进入 `P1`：

- [ ] 6 个 `ModelOpt` baseline case 仍可作为回归基线。
- [x] 与 quant config、mixed export、metadata、offload/fallback 相关的测试不回退。
- [x] 文档、测试、代码对“当前支持范围”的表述一致。
- [x] `P0` 没有偷偷引入 `mxfp4` 半成品接口。

### P0 Evidence

- [x] Audit note 路径：`artifacts/diffusion_mxfp4/plan_notes/p0-audit-note-2026-05-26.md`
- [x] 测试结果路径：`artifacts/diffusion_mxfp4/plan_notes/p0-test-attempt-2026-05-26.md`
- [x] Server baseline follow-up 路径：`artifacts/diffusion_mxfp4/plan_notes/p0-server-baseline-followup-2026-05-27.md`
- [x] 提交 / diff 说明：补充了 P0 单测、修正文档支持边界，修复了 `FP4` alias 解析与 FLUX.2 mixed-NVFP4 directory override guardrail，并把若干 optional dependency / cache import 路径改成 lazy 或 fail-late 以便 `P0` unit tests 在 `exp_env` 中可运行。

---

## P1 Checklist: Diffusion `mxfp4` Feasibility Spike

`P1` 的目标只有一个：回答“diffusion DiT dense linear 的 `mxfp4` backend 是否值得继续推进”。

### P1.1 Kickoff Decisions

开始实现前必须先冻结：

- [ ] internal family name
  - 推荐：`mxfp4_dit_blackwell`
- [ ] 首个验证模型
  - 优先：`FLUX.1`
  - 备选：`FLUX.2`
- [ ] 是否延后 packed QKV
- [ ] 首版 checkpoint 只支持 `--transformer-path` 还是同时支持 `--transformer-weights-path`
- [ ] representative shape set 的来源模型
- [ ] trajectory similarity 阈值冻结流程

如果以上任一项未定，暂停编码。

### P1.2 Freeze Checkpoint Contract

- [ ] 冻结 internal family name。
- [ ] 冻结最小 on-disk schema。
- [ ] 冻结 discovery order：
  - `--transformer-path/config.json`
  - `--transformer-weights-path` metadata
  - 否则 hard fail
- [ ] 冻结 fail policy：
  - metadata 缺失时不自动猜
  - base model config 不倒推 `mxfp4`
  - 首版不做 silent fallback
- [ ] 明确 BF16 fallback 层记录方式。
- [ ] 明确 packed-QKV 在 `P1` 是否支持。

输出物：

- [ ] `checkpoint_contract.md` 或等价文档
- [ ] 示例 `config.json` / schema 片段

### P1.3 Freeze Representative Shape Set

- [ ] 从目标 `FLUX` checkpoint 中提取代表性 shape。
- [ ] 覆盖以下类型：
  - attention projection
  - attention out projection
  - MLP expansion projection
  - MLP down projection
  - padding / alignment regression
- [ ] 把 exact `(M, N, K)` 清单记录进 repo。
- [ ] 确保 correctness 与 profiling 使用同一组 shape。

输出物：

- [ ] `representative_shapes.md` 或等价记录
- [ ] 如果有，shape 抽取脚本路径

### P1.4 Minimal Runtime Path

- [ ] 新 family 能被 diffusion quant registry 识别。
- [ ] quant config class 能解析冻结后的 checkpoint contract。
- [ ] loader 能按 discovery order 正确工作。
- [ ] dense-linear quant method 至少能接入：
  - `ReplicatedLinear`
  - `ColumnParallelLinear`
  - `RowParallelLinear`
- [ ] 权重后处理路径可复现。
- [ ] 首版不支持的行为必须 hard fail，而不是 silent fallback。

### P1.5 Synthetic Correctness

- [ ] 为 representative shape set 建 synthetic correctness test。
- [ ] 建权重打包 / 后处理 correctness test。
- [ ] 与 dequant ref 或高精度 ref 做数值对比。
- [ ] 结果必须可复现。

建议输出：

- [ ] correctness 测试文件路径
- [ ] correctness 结果日志路径

### P1.6 Backend Viability / Profiling

- [ ] 对 representative shape set 跑 microbenchmark / profiler。
- [ ] 记录候选 backend 选择依据。
- [ ] 证明所选 backend 不只是“能跑”，而是对目标 shape 合理。
- [ ] 如存在明显不适配 shape，明确记录。

输出物：

- [ ] profiling 报告路径
- [ ] backend choice note 路径
- [ ] 至少一份 profiler trace 路径

### P1.7 Trajectory Similarity Preparation

- [ ] 先采集 `BF16 vs BF16` 重复运行噪声基线。
- [ ] 再采集现有 diffusion `NVFP4` 参考分布。
- [ ] 基于这两组数据冻结 `MXFP4` trajectory similarity 阈值。
- [ ] 在阈值冻结前，不允许写“已通过，只是感觉差不多”。

输出物：

- [ ] BF16 baseline 报告路径
- [ ] NVFP4 reference 报告路径
- [ ] threshold freeze note 路径

### P1.8 Trajectory Similarity Validation

- [ ] 使用固定 prompt / seed / resolution / step count。
- [ ] 运行 BF16 vs candidate `mxfp4` 对比。
- [ ] 记录每步 latent cosine / error。
- [ ] 依据冻结后的阈值判断 pass / fail。

输出物：

- [ ] trajectory comparison JSON / report 路径

### P1.9 End-To-End Smoke

- [ ] 运行至少一次固定模型的 e2e smoke。
- [ ] 记录：
  - 使用的 checkpoint
  - prompt
  - seed
  - resolution
  - steps
  - 是否成功生成
- [ ] 如失败，记录失败类型：
  - loader/config 问题
  - kernel/backend 问题
  - quality / stability 问题

### P1.10 Minimal Performance Evidence

- [ ] 记录 BF16 vs `mxfp4` 的 peak GPU memory。
- [ ] 记录固定 case 的 e2e latency。
- [ ] 记录 representative shape microbenchmark 数据。
- [ ] 没有数据时，不写任何性能收益结论。

### P1.11 P1 Go / No-Go Decision

全部满足才允许进入 `P2`：

- [ ] synthetic dense-linear correctness test 通过
- [ ] representative shape profiling 能支持 backend 决策
- [ ] trajectory similarity 通过已冻结阈值
- [ ] 至少一次 e2e smoke 成功
- [ ] 实现复杂度没有明显失控

任一条件不满足则：

- [ ] 记录为 `NO_GO`
- [ ] 输出阻塞结论
- [ ] 不引入半成品主线接口

### P1 Evidence

- [ ] checkpoint contract 路径：
- [ ] representative shape 清单路径：
- [ ] correctness 结果路径：
- [ ] profiling 报告路径：
- [ ] backend choice note 路径：
- [ ] trajectory 阈值记录路径：
- [ ] trajectory 结果路径：
- [ ] e2e smoke 结果路径：
- [ ] go / no-go 结论路径：

---

## P2 Checklist: Formal Diffusion `mxfp4` MVP

只有在 `P1` 明确 `GO` 后才能开始。

### P2.1 Scope Reconfirm

- [ ] 仍然限定为：
  - `Blackwell CUDA`
  - `FLUX`
  - 单 DiT
  - pre-quantized transformer override
- [ ] 不偷偷扩到多模型、多平台、多 backend。

### P2.2 Productize Runtime Path

- [ ] 把 `P1` prototype 收敛成正式 runtime 路径。
- [ ] quant registry 接入正式 family。
- [ ] quant config class 收敛。
- [ ] loader / post-load processing 收敛。
- [ ] 不支持的路径保持显式 fail。

### P2.3 Automated Tests

- [ ] synthetic correctness 接入 pre-merge lane。
- [ ] 至少一条 diffusion e2e smoke 接入 pre-merge lane。
- [ ] trajectory similarity 阈值检查具备自动化落点，或明确说明为何尚不能 gate。

### P2.4 CI Lane Binding

- [ ] synthetic correctness 对齐 `B200` kernel/unit 风格。
- [ ] e2e smoke 接入现有 diffusion `B200` lane，或新增明确 lane。
- [ ] 如果检查暂时是 non-gating artifact，也要明确归档位置。

### P2.5 Performance / Resource Report

- [ ] 输出最小性能报告：
  - peak GPU memory
  - fixed-case e2e latency
  - representative shape profiling
- [ ] 不夸大收益；只陈述已有证据。

### P2.6 Docs

- [ ] 只有当实现、测试、CI、证据都齐备时，才更新正式文档。
- [ ] 文档必须准确描述：
  - 支持平台
  - 支持模型范围
  - 支持输入工件形式
  - 不支持项

### P2 Exit Gate

- [ ] 运行路径 productized
- [ ] 自动化测试具备明确落点
- [ ] 至少一条 e2e smoke 是稳定的
- [ ] trajectory 阈值有明确 gate 或明确 blocker 说明
- [ ] 性能 / 资源报告齐备
- [ ] 文档与代码一致

### P2 Evidence

- [ ] 代码路径：
- [ ] CI lane / suite：
- [ ] 测试结果路径：
- [ ] benchmark / profile 路径：
- [ ] 文档更新路径：

---

## Session Handoff Checklist

每次 session 结束前必须更新：

- [x] 本次完成了哪些 checklist 项
- [x] 哪些项仍然 `IN_PROGRESS`
- [x] 哪些项被判定为 `BLOCKED`
- [x] 新增了哪些 artifact 路径
- [x] 新发现的风险 / open questions
- [x] 下一 session 最应该先做的 1 到 3 件事

建议 handoff 模板：

```md
### Session Handoff

- Completed:
- In progress:
- Blocked:
- Evidence:
- Open questions:
- Next recommended steps:
```

### Session Handoff

- Completed:
  - 完成 P0 代码审计并输出 audit note
  - 补齐 `transformer-path` / `transformer-weights-path` 优先级与 NVFP4 aggregation 相关单测
  - 修正文档中的优先级语义、ModelOpt smoke/regression 覆盖表述、NVFP4 backend env alias
  - 修复 `FP4` alias config 解析与 FLUX.2 mixed NVFP4 directory override guardrail
  - 解决目标 `P0` unit tests 的 import / cache 环境阻塞，使其在 `exp_env` 中通过
  - 进一步收紧了 server-side collect/run 的 import-time optional dependency：
    `torchcodec` video decoder eager import、`registry <-> pipelines_core`
    circular import、`cache_dit` hard dependency、`stages.__init__` 对 3D/AV
    stage 的 eager import
- In progress:
  - Hugging Face-only `BBuf/*-modelopt-*` baseline 资产下载链路诊断中；
    当前已定位到 Xet backend / proxy 兼容性，而不是普通 metadata 可达性
- Blocked:
  - 6 个 ModelOpt baseline 需要的 9 个 repo/checkpoint 均不在本地 cache
  - 官方 ModelScope repo 可以继续下载，但 Hugging Face-only `BBuf/*-modelopt-*`
    仍然受 Xet backend / proxy 兼容性影响，因此无法宣称 baseline 已可运行
- Evidence:
  - `artifacts/diffusion_mxfp4/plan_notes/p0-audit-note-2026-05-26.md`
  - `artifacts/diffusion_mxfp4/plan_notes/p0-test-attempt-2026-05-26.md`
  - `artifacts/diffusion_mxfp4/plan_notes/p0-server-baseline-followup-2026-05-27.md`
- Open questions:
  - 是否要在 `P0` 进一步加入 `--transformer-path` 与 `--transformer-weights-path` quant family mismatch 的 hard check
  - 是否要在 test harness / test data 定义层避免 import-time 触发远端模型解析，以便 server tests 支持真正的 `collect-only`
  - 当前最有希望的 workaround 是手调 `hf_xet` reconstruction window；是否应将其固化成临时下载脚本，而不是继续依赖 `huggingface-cli`
- Next recommended steps:
  - 继续验证手调 `hf_xet` 小窗口配置是否能完整 materialize 一个 shard
  - 若验证成功，把同样参数用于 `flux1 fp8` / `flux1 nvfp4` quant repo 正式下载
  - quant repo 完成后，优先跑 `flux1_modelopt_fp8_t2i` 的 targeted single-case smoke

## 最终完成定义

只有满足以下条件，整个任务才能被认为完成：

- [ ] `P0` 完成
- [ ] `P1` 给出清晰 `GO` 结论
- [ ] `P2` 完成并有自动化落点
- [ ] 文档、代码、测试、CI、性能证据一致
- [ ] 没有把半成品 `mxfp4` 支持暴露给用户

如果最终路线不可行，也可以视为“任务正确结束”，前提是：

- [ ] 已完成 `P0`
- [ ] 已完成 `P1` 验证
- [ ] 已输出明确 `NO_GO` 结论
- [ ] 已记录阻塞原因
- [ ] 没有提交误导性接口
