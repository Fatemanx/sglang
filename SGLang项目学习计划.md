# SGLang 项目学习计划（性能优化方向）

## Context

SGLang 是一个高性能 LLM 推理服务框架，包含两大核心：
1. **前端 DSL** — 结构化生成的编程语言
2. **SRT 运行时** — 高性能推理引擎（调度、KV Cache、分布式、投机解码等）

**学习者背景：** 了解 Transformer 架构和推理流程，未深入过 serving 框架源码。
**学习目标：** 理解 SGLang 的性能优化手段（调度策略、内核优化、投机解码、内存管理等）。

本计划从外到内、从简到难，侧重性能相关的设计决策和实现细节。

---

## Phase 1: 跑通 & 建立直觉（1-2 天）

**目标：** 能启动服务、发请求、跑 benchmark，建立对系统行为的直觉。

### 1.1 环境搭建
- 阅读 [docs/get_started/](docs/get_started/) 安装文档
- 安装 SGLang 并确认 GPU 可用

### 1.2 启动服务 & 发请求

**推荐用小模型快速验证**（按显存从小到大）：

| 模型 | 显存需求 | 适用场景 |
|------|---------|---------|
| `meta-llama/Llama-3.2-1B-Instruct` | ~3GB | 默认推荐，验证基础流程 |
| `Qwen/Qwen2.5-1.5B-Instruct` | ~4GB | 中文场景验证 |
| `Qwen/Qwen3-1.7B-FP8` | ~2GB | 验证 FP8 量化路径 |
| `meta-llama/Llama-3.2-3B-Instruct` | ~7GB | 显存充裕时用 |
| `RedHatAI/Llama-3.2-3B-quantized.w8a8` | ~4GB | 验证 W8A8 量化路径 |

启动命令示例：
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --port 30000 \
    --mem-fraction-static 0.8
```

- 用 curl 或 Python 调用 OpenAI 兼容 API (`/v1/chat/completions`)
- 观察日志输出，理解启动流程（KV Cache 大小、batch size 等关键信息）

> 说明：`python/sglang/test/test_utils.py` 中定义了 `DEFAULT_SMALL_MODEL_NAME_FOR_TEST` 等常量，CI 测试默认就用这些小模型。

### 1.3 前端 DSL 体验

**SGLang 核心原语速览**（参考 [examples/frontend_language/usage/readme_examples.py](examples/frontend_language/usage/readme_examples.py)）：

#### `@sgl.function` — 程序定义装饰器
将一个 Python 函数包装成 SGL 程序。第一个参数 `s` 是 ProgramState，承载执行上下文。

```python
@sgl.function
def text_qa(s, question):
    s += "Q: " + question + "\n"          # 拼接 prompt
    s += "A:" + sgl.gen("answer", stop="\n")  # 触发生成

state = text_qa.run(question="What is the capital of France?")
print(state["answer"])  # 取出生成结果
```

#### `sgl.gen()` — 文本生成原语
触发 LLM 生成，结果存入 state 的命名变量。

```python
sgl.gen("name", max_tokens=64, temperature=0.7, stop="\n")
sgl.gen("answer", regex=r"\d+")          # 正则约束
sgl.gen("data", json_schema=schema_str)  # JSON Schema 约束
sgl.gen("tool", choices=["search", "calc"])  # 选择生成
```

常用参数：`max_tokens`、`temperature`、`top_p`、`top_k`、`stop`、`regex`、`json_schema`、`choices`。

#### `sgl.select()` — 受限选择
从预定义选项中根据 logprob 选最可能的。

```python
s += "Sentiment: " + sgl.select("sentiment", choices=["positive", "negative", "neutral"])
```

#### 角色标记 — `system()` / `user()` / `assistant()`
用于多轮对话场景，自动套用 chat template。

```python
@sgl.function
def chat(s, q):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(q)
    s += sgl.assistant(sgl.gen("reply", max_tokens=256))
```

#### `s.fork(n)` — 并行分支
创建 n 个独立分支，可并行生成后再合并。共享前缀会被自动复用（RadixAttention 的关键应用）。

```python
forks = s.fork(2)
for i, f in enumerate(forks):
    f += f"Tip {i+1}: " + sgl.gen("detail", max_tokens=128)
```

#### `run_batch()` — 批量执行
同一程序对多个输入并行执行，自动利用 continuous batching。

```python
states = text_qa.run_batch(
    [{"question": q} for q in questions],
    num_threads=8,
    progress_bar=True,
)
```

#### 流式输出 — `stream=True`
逐 token 返回结果，配合 `state.text_iter()` 使用。

```python
state = text_qa.run(question="...", stream=True)
for chunk in state.text_iter():
    print(chunk, end="", flush=True)
```

### 实操步骤
1. 启动 SGLang server（用 1.2 的小模型）
2. 运行 [examples/frontend_language/usage/readme_examples.py](examples/frontend_language/usage/readme_examples.py)
3. 修改示例：用 `s.fork()` 实现并行思考、用 `regex` 约束输出格式
4. 理解 SGLang 程序 = Python 函数 + 生成原语 + 自动前缀复用

### 1.4 跑一个 Benchmark
- 运行 [benchmark/gsm8k/bench_sglang.py](benchmark/gsm8k/bench_sglang.py) 感受批量推理

---

## Phase 2: 前端语言层（1-2 天，快速过）

**目标：** 理解 SGLang DSL 的设计思想，重点关注与性能相关的前缀缓存机制。

### 关键文件
| 文件 | 作用 |
|------|------|
| [python/sglang/lang/ir.py](python/sglang/lang/ir.py) | IR 节点定义 |
| [python/sglang/lang/tracer.py](python/sglang/lang/tracer.py) | 前缀提取（性能关键） |
| [python/sglang/lang/interpreter.py](python/sglang/lang/interpreter.py) | 程序执行引擎 |

### 学习路径
1. 快速浏览 `api.py` 和 `ir.py`，理解 DSL 的抽象层次
2. **重点读 `tracer.py`** — 理解如何通过静态分析提取公共前缀，这是 RadixAttention 的前端基础
3. 理解 `run_batch()` 的并发模型（线程池 + 异步 IO）

---

## Phase 3: 服务架构 — 进程模型 & 请求流（3-5 天）

**目标：** 理解 SRT 的多进程架构和请求从进入到返回的完整路径。

### 架构概览
```
HTTP Server (FastAPI)
    ↓
TokenizerManager (主进程)
    ↓ ZMQ IPC
Scheduler (子进程) ← 核心调度逻辑
    ↓
ModelRunner (在 Scheduler 进程内)
    ↓ ZMQ IPC
DetokenizerManager (子进程)
    ↓
HTTP Response
```

### 关键文件
| 文件 | 作用 |
|------|------|
| [python/sglang/srt/entrypoints/engine.py](python/sglang/srt/entrypoints/engine.py) | Engine 类，进程编排 |
| [python/sglang/srt/entrypoints/http_server.py](python/sglang/srt/entrypoints/http_server.py) | HTTP API 路由 |
| [python/sglang/srt/managers/tokenizer_manager.py](python/sglang/srt/managers/tokenizer_manager.py) | 输入 tokenize |
| [python/sglang/srt/managers/scheduler.py](python/sglang/srt/managers/scheduler.py) | 调度器（3800+ 行，核心） |
| [python/sglang/srt/managers/detokenizer_manager.py](python/sglang/srt/managers/detokenizer_manager.py) | 输出 detokenize |
| [python/sglang/srt/managers/io_struct.py](python/sglang/srt/managers/io_struct.py) | 进程间通信数据结构 |

### 学习路径
1. 从 `engine.py` 的 `Engine.__init__()` 开始，看三个进程如何启动
2. 跟踪一个请求：HTTP → TokenizerManager → Scheduler → ModelRunner → Detokenizer → Response
3. 重点读 `scheduler.py` 的主循环（`event_loop_normal`），理解 batch 调度策略
4. 读 `io_struct.py`，理解进程间传递的数据结构

---

## Phase 4: 模型执行 & KV Cache（3-5 天）

**目标：** 理解模型如何加载、forward pass 如何执行、KV Cache 如何管理。

### 关键文件
| 文件 | 作用 |
|------|------|
| [python/sglang/srt/model_executor/model_runner.py](python/sglang/srt/model_executor/model_runner.py) | 模型加载 & forward |
| [python/sglang/srt/model_executor/forward_batch_info.py](python/sglang/srt/model_executor/forward_batch_info.py) | Forward pass 输入结构 |
| [python/sglang/srt/managers/schedule_batch.py](python/sglang/srt/managers/schedule_batch.py) | Req & ScheduleBatch 定义 |
| [python/sglang/srt/mem_cache/](python/sglang/srt/mem_cache/) | KV Cache 内存管理 |
| [python/sglang/srt/layers/attention/](python/sglang/srt/layers/attention/) | Attention 后端 |

### 学习路径
1. 读 `model_runner.py` 的模型加载流程（`load_model()`）
2. 理解 `forward_decode()` vs `forward_extend()` 的区别（decode = 逐 token，extend = prefill）
3. 读 `schedule_batch.py` 的 `Req` 类，理解请求的生命周期状态
4. 读 `mem_cache/memory_pool.py`，理解 KV Cache 的分配与回收
5. 了解 RadixCache（前缀缓存）的原理

---

## Phase 5: 性能优化核心专题（重点，每个 3-5 天）

### 5.1 投机解码 (Speculative Decoding) — 推理加速核心
- 目录：[python/sglang/srt/speculative/](python/sglang/srt/speculative/)
- 理解 draft-verify 流程、bonus token 机制
- 注意 `.claude/rules/speculative-naming.md` 中的命名规范
- 关注 accept rate 和 accept length 的计算

### 5.2 CUDA 内核优化 (sgl-kernel)
- 目录：[sgl-kernel/](sgl-kernel/) — AOT 编译内核
- 目录：[python/sglang/jit_kernel/](python/sglang/jit_kernel/) — JIT Triton 内核
- 关注 FlashInfer attention 后端、fused kernels
- 用 `ncu` (Nsight Compute) 分析内核性能

### 5.3 CUDA Graph & 编译优化
- 目录：[python/sglang/srt/model_executor/cuda_graph_runner.py](python/sglang/srt/model_executor/cuda_graph_runner.py)
- 理解 CUDA Graph capture/replay 如何消除 kernel launch overhead
- PyTorch compile 集成：[python/sglang/srt/compilation/](python/sglang/srt/compilation/)

### 5.4 分布式推理 & 通信优化
- 目录：[python/sglang/srt/distributed/](python/sglang/srt/distributed/)
- 理解 TP（张量并行）的 all-reduce 开销
- 计算-通信 overlap 策略

### 5.5 Prefill-Decode 分离 (Disaggregation)
- 目录：[python/sglang/srt/disaggregation/](python/sglang/srt/disaggregation/)
- 理解为什么分离 prefill 和 decode 能提升吞吐

### 5.6 Profiling 实战
- 使用 `--enable-torch-profile` 生成 trace
- 用 Chrome trace viewer (chrome://tracing) 分析执行时序
- 识别 GPU idle、kernel launch overhead、通信瓶颈
- 对比 SGLang vs vLLM 的 kernel 选择差异

---

## Phase 6: 测试 & 贡献（持续）

### 测试体系
- [test/](test/) — 集成测试（按 CI stage 组织）
- [python/sglang/test/](python/sglang/test/) — 测试工具和 runner
- 先读 `test/README.md` 了解 CI 布局

### 贡献流程
- 阅读 [docs/developer_guide/](docs/developer_guide/)
- 从小 issue 或 benchmark 改进入手
- 跑 `python -m pytest test/srt/test_xxx.py` 验证改动

---

## 推荐学习策略

1. **先跑后读** — 每个阶段先把代码跑起来，观察行为，再读源码
2. **加日志追踪** — 在关键路径加 `print` 或 `logger.info` 追踪请求流
3. **用 profiler** — `--enable-torch-profile` 生成 trace，用 Chrome trace viewer 看执行时序
4. **对比 vLLM** — SGLang 很多设计是对 vLLM 的改进，对比阅读能加深理解
5. **读 PR 历史** — 重要特性的 PR 通常有详细的设计讨论和 benchmark 数据
6. **跑 benchmark 对比** — 修改参数（batch size、mem-fraction-static、chunk-prefill-size）观察性能变化

---

## 性能优化关键概念速查

| 概念 | SGLang 实现 | 性能收益 |
|------|-------------|----------|
| RadixAttention | `mem_cache/radix_cache.py` | 前缀复用，减少重复计算 |
| Continuous Batching | `scheduler.py` 主循环 | 提升 GPU 利用率 |
| Chunked Prefill | `model_runner.py` `forward_split_prefill` | 长序列不阻塞 decode |
| CUDA Graph | `cuda_graph_runner.py` | 消除 kernel launch overhead |
| FlashInfer | `layers/attention/` | 高效 attention 内核 |
| Speculative Decoding | `speculative/` | 减少 decode 步数 |
| KV Cache 量化 | `mem_cache/` | 减少显存占用 |
| Overlap | scheduler mixins | 计算-通信并行 |

---

## 验证方式

每个阶段完成后，尝试回答以下问题来检验理解：

- Phase 1: SGLang 启动时 `mem_fraction_static` 参数如何影响 KV Cache 容量？
- Phase 2: 前缀缓存（RadixAttention）的前端静态分析如何工作？
- Phase 3: Scheduler 如何决定哪些请求进入当前 batch？Continuous batching 的退出条件是什么？
- Phase 4: CUDA Graph 在什么条件下 capture？为什么 decode 适合用 CUDA Graph 而 prefill 不适合？
- Phase 5: 投机解码的 accept rate 受哪些因素影响？如何通过 profiling 定位性能瓶颈？
