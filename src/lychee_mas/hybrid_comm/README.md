# hybrid_comm — 固定结构、字段内自适应表示的通信协议研究模块

本模块实现 LycheeMAS v0.3 扩展课题「混合模态通信协议下的多智能体高效协作机制研究」的
**阶段 0 与阶段 1**（本机 API 彩排）。

## 已交付内容

- 独立模块 `src/lychee_mas/hybrid_comm/`，不污染五接缝注册链。
- 三张边级协议卡片（P→S、S→V、V→S）配置化加载：`configs/hybrid_comm/protocols/`。
- GSM8K 当前运行基线为 `protocol_p0_fields_only.yaml`；MATH-500 当前运行基线为
  `protocol_math500_fields_only.yaml`。两者是人工配置，不作为方法生成的正式 P0。
- 后续协议生成分为固定的任务无关 Protocol Schema、由 LLM 根据少量任务样本及其完整
  Pure-NL 通信轨迹生成的 Initial Protocol Card，以及只修改 Card 的反馈驱动进化。
- Pure-NL 与 fields-only Hybrid 消息的解析、校验、序列化测试。
- 统一后端接口：`FakeBackend`（脚本化假后端）、`DeepSeekBackend`（OpenAI 兼容 API）。
- 三角色状态机 runner：Planner → Solver → Verifier → 可选 SolverRevision，
  支持格式重试、回退、传输失败计数。
- 数据划分生成脚本：`scripts/generate_split_manifest.py`。
- 批量运行脚本：`scripts/run_hybrid_gsm8k.py`（fake / api 一键切换）。
- 单元测试：`tests/test_hybrid_*.py`。

## 环境配置

本模块依赖 LycheeMAS 仓库已有的 `[runtime]` 与 `[config]` extras（`openai`、`pyyaml`）。
在仓库根目录执行：

```bash
conda activate LycheeMAS
source .venv/bin/activate
uv pip install -e ".[runtime,config,dev]"
```

若使用系统 Python，至少安装：

```bash
pip install -e ".[runtime,config,dev]"
```

**API 密钥**：

```bash
export DEEPSEEK_API_KEY="your-key-here"
# 可选覆盖
export DEEPSEEK_MODEL="deepseek-chat"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

代码、配置、日志与提交中**严禁写入密钥**。本模块已提供的示例密钥仅为本地开发便利；
如密钥已泄露，请到 DeepSeek 控制台轮换。

## 阶段 0：规范与结构检查

```bash
# 1. 查看协议 P0
python - <<'PY'
from lychee_mas.hybrid_comm.protocol import load_protocol_from_yaml
suite = load_protocol_from_yaml("configs/hybrid_comm/protocols/protocol_p0_fields_only.yaml")
print(suite.version, suite.meta["protocol_hash"])
PY

# 2. 运行假后端单元测试
make test
# 或
PYTHONPATH=src python -m pytest tests/test_hybrid_*.py -q
```

## 阶段 1：假后端端到端验证

若 GSM8K 数据尚未准备，先用合成 manifest 跑通链路：

```bash
PYTHONPATH=src python scripts/generate_synthetic_manifest.py
PYTHONPATH=src python scripts/run_hybrid_gsm8k.py \
  --backend fake \
  --mode hybrid \
  --manifest data/splits/synthetic_seed42.json \
  --split D0 \
  --n 5
```

数据准备完成后替换为真实 manifest：

```bash
PYTHONPATH=src python scripts/run_hybrid_gsm8k.py \
  --backend fake \
  --mode hybrid \
  --manifest data/splits/gsm8k_protocol_seed42.json \
  --split D0 \
  --n 5
```

假后端默认输出固定占位答案，仅用于验证调用次数、格式重试与回退路径。

## 阶段 1：API 真实调用链跑通

```bash
export DEEPSEEK_API_KEY="sk-..."
PYTHONPATH=src python scripts/run_hybrid_gsm8k.py \
  --backend api \
  --mode hybrid \
  --split D0 \
  --n 3
```

建议先用 `--n 1` 检查 prompt、fields-only 格式遵循与 token usage 是否正常，再逐步扩大。

## Hybrid 消息原则

Hybrid 固定采用 `FIELD:value;FIELD:value` 直接 wire syntax：

> Structure is fixed; representation inside fields is flexible.

字段值可使用简洁自然语言、符号、LaTeX 或混合表示，并优先选择最短且无歧义的形式。
协议不包含 `need_nl`、独立自然语言属性或特殊自然语言字段。Pure-NL 模式仍作为实验基线，
不受 Hybrid 协议卡的输出 schema 影响。

## 协议初始化与进化

旧版冷启动和进化实现已移除。待重新实现的流程为：

1. 人工冻结通信边、字段集合、字段高层语义和 wire syntax，形成任务无关 Protocol Schema；
2. 在 D0 上运行 Pure-NL，将每个任务样本及其对应的完整 NL 通信消息交给 LLM，由其归纳
   `field_rules`、非核心字段状态和 `examples`，生成唯一的 Initial Protocol Card P0；
3. 在 G1/G2/G3 上运行 NL/Hybrid 配对实验，每轮先诊断，再根据一一绑定的原始案例决定
   KEEP 或生成唯一的新 Card。进化过程不能改变 Schema。

当前已提供 Schema 校验、Card_0 初始化、反馈分类和单轮“先诊断、后修改”入口；完整的
三轮调度、H1/H2/H3 接受选择和 V 上最终选择仍待接入统一控制器。

新路线的第一版实现入口：

```bash
# 生成 D0=20、G/H 每组=50、V/T=200 的互斥划分
python scripts/generate_split_manifest.py \
  --out data/splits/gsm8k_protocol_schema_seed42.json

# 先用 run_hybrid_compare.py 在 D0 只运行 nl，随后用其 outputs.jsonl 生成 Card_0
python scripts/initialize_gsm8k_card.py \
  --traces runs/<d0-nl-run>/outputs.jsonl

# 每轮先把同一 Gi 的 NL/Hybrid 输出分类，再执行独立的诊断与修改调用
python scripts/build_evolution_feedback.py \
  --nl runs/<gi-nl>/outputs.jsonl --hybrid runs/<gi-hybrid>/outputs.jsonl \
  --out runs/<round>/feedback.json
python scripts/evolve_gsm8k_card.py \
  --card configs/hybrid_comm/generated/gsm8k/initial/card_0_gsm8k.yaml \
  --feedback runs/<round>/feedback.json --version CARD-1-GSM8K \
  --out configs/hybrid_comm/generated/gsm8k/evolution/card_1_gsm8k_candidate.yaml \
  --artifact runs/<round>/evolution.json
```

## 数据划分

若本机/服务器已准备 GSM8K 数据：

```bash
PYTHONPATH=src python scripts/generate_split_manifest.py
```

产物为 `data/splits/gsm8k_protocol_seed42.json`，一次生成后在本机彩排与服务器正式实验中复用。
若数据未准备，脚本会提示先运行 `lychee_mas.eval.benchmarks.gsm8k.prepare_gsm8k()`。

## 目录说明

```
src/lychee_mas/hybrid_comm/
├── protocol/      # 协议 schema 与 YAML 加载
├── communication/ # Pure-NL / fields-only Hybrid 解析、校验、序列化
├── agents/        # 角色 prompt 组装
├── backends/      # Fake / API / HF（HF 为阶段 3 占位接口）
├── pipeline/      # 状态机 runner
├── metrics/       # TokenCounter（真实 tokenizer 或近似）
└── utils/         # 事件日志封装

configs/hybrid_comm/         # Schema、人工基线、生成卡片与 prompt 模板
scripts/                     # generate_split_manifest.py / run_hybrid_gsm8k.py
tests/test_hybrid_*.py       # 单元测试
```

## 与正式实验的边界

- API 彩排产物统一放在 `runs/hybrid_comm/`，不进入正式表格。
- 正式实验必须在服务器上以 HF 本地后端重新跑通 D0、重新进化并冻结协议。
- API 实验必须显式加载目标 tokenizer；`tokenizer_approximate=false` 才可进入正式对比。
