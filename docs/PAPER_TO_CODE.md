# 论文实验与代码对应说明

本说明依据 `HASI_KBS_20260920_FULL.zip` 内的 `main.tex`。当前论文的正式实验为 Hetionet 上的实体类型分类（node）和 Compound–treats–Disease 关系预测（relation），入口为 `experiments/knowledge_state_maintenance/`。历史 PubMed、PrimeKG、旧 HASI 消融及 `supplementary_experiments/` 批次不属于这版论文的结果来源。

本包只包含代码、协议、配置和说明。下文提到的 `data/`、模型、预测数组、统计 CSV、审计 JSON 和图表都是读者准备数据并运行后生成的文件，压缩包不携带这些文件。仅有代码不能直接重放原论文结果；数据准备见 `DATA_PREPARATION.md`，执行入口与环境安装以包根 README 为准。

## 1. 实验实现

| 论文内容 | 代码位置 | 关键行为 |
| --- | --- | --- |
| 数据构建与撤回语义 | `experiments/knowledge_state_maintenance/runtime.py` 的 `prepare_source`、`prepare_split`、`load_data`、`withdraw` | 节点任务读取预处理的无来源特征缓存；关系任务读取 Hetionet 原始 JSON，排除 held-out/withdrawn 关系泄漏；撤回关系不会转为负例。 |
| GCN、优化、验证最佳 checkpoint | 同文件的 `new_model`、`fit` | 两层 GCN；维护完整执行 150 步，保存 25/50/100/150 步固定与验证最佳快照；base 和 Matched-Retrain 最多 300 步、patience 50。 |
| 参考评分与选择 | 同文件的 `compute_components`、`select_refs`，以及 `src/hasi/hub_identification/` | Full、Structure、Random-all、Random-train、Class-balanced-train 的参考政策；参考组配额和训练标签范围检查。 |
| 参考表示约束 | `src/hasi/anchor_stabilization/anchor_loss.py`，runtime 的 `fit` | Primary/Secondary 权重与乘子；原输入与撤回后输入的目标快照。 |
| 开发选参和样本量 | `experiments/knowledge_state_maintenance/run.py` 的 `grid`（由 runtime 导入）、`freeze`、`select_and_release` | 两个 development splits、每族九候选、验证质量选参；正式配置和 base 数冻结。 |
| 正式配对实验 | 同文件的 `formal_specs`、`worker`、`execute` | Original、Input-only、Continue-FT、Fixed-FT、Tuned-FT、Matched-Retrain、参考方法、强度干预、节点覆盖比较，以及指定请求的附加重训。 |
| 分层汇总和配对区间 | `experiments/knowledge_state_maintenance/report.py` | 参考重复 → 请求 → base → split；正式统计单位为五个 split 均值，输出样本 SD 和配对 t 区间。 |

论文中的 `FT`、`Full-old`、`Full-input`、`Random-all`、`Structure` 在调参比较表中分别对应结果字段 `Tuned-FT`、`Tuned-Full-old`、`Tuned-Full-input`、`Tuned-Random-all`、`Tuned-Structure`。固定机制实验必须筛选 `mode=fixed`、`budget=150`；调参主结果筛选 `mode=selected`、`budget=150`，不能把二者混合。

## 2. 数据、训练、统计的生成顺序

1. 准备 Hetionet 原始图和节点预处理缓存，核对输入身份与特征构建协议。
2. 初始化运行环境并运行实现测试；GPU preflight 使用真实完整节点图检查零权重约束与 FT 的数值一致性。
3. 选择复现方式。`redevelop` 重新运行开发选参及样本量计算；`paper` 使用论文记录的配置与 seeds 直接重建正式试验计划。前者可能因环境数值差异选出不同配置或 base 数，后者不产生新的开发搜索成本。
4. 执行正式 base、每 base 两个独立请求和方法组合。每个请求都从对应 base 出发，不能将前一个请求更新后的模型接着用于下一个请求。
5. 正式运行完整后执行 `report`，从每次运行保存的预测概率重新计算质量指标，并生成 `reproduction/tables/`。
6. 执行 `analysis`，由新写的 `scripts/summarize_paper.py` 读取本次生成的正式统计表，导出论文主要比较、机制、类别、关系和计时摘要。

从包根目录运行下面命令可查看计划；去掉 `--dry-run` 后才执行完整流程。数据必须事先准备好。三卡配置可将 `--devices` 改为 `cuda:0,cuda:1,cuda:2`。

```bash
python scripts/reproduce.py --stage all --protocol paper --devices cuda:0 --dry-run
python scripts/reproduce.py --stage all --protocol redevelop --devices cuda:0 --dry-run
```

包内 `python scripts/reproduce.py --help` 列出各阶段的调用方式。正式汇总的底层入口为：

```bash
python -m experiments.knowledge_state_maintenance.run report
```

这个命令需要已完成的正式运行、数据缓存、release 与 source hash，不能对空代码包直接运行。训练输出保存在包内 `reproduction/` 下，执行前应预留足够存储空间。

## 3. 正文和附录表格对应

以下使用论文 LaTeX 标签标识表格，避免排版导致表号变化。

| 论文表格或结果 | 原始统计来源（均在运行后生成） | 筛选/汇总方式 |
| --- | --- | --- |
| 主比较 `tab:maintenance-main` | `tables/metrics_summary.csv`、`metrics_per_split.csv` | Tuned 方法的 selected/150，加 Input-only 的 selected/0、Matched-Retrain 的 selected/300；`quality`、`retrain_js`、`first_use_s`。 |
| 配对结果 `tab:maintenance-paired` | `tables/paired_intervals.csv`、`metrics_per_split.csv` | 包含 FT−Input-only、各参考方法−FT 等对比；逐 split 配对后计算区间，不能相减两个非配对 CI。 |
| 撤回与继续训练交互 `tab:maintenance-interaction` | `tables/withdrawal_interaction.csv` | `(Fixed-FT−Input-only)−(Continue-FT−Original)`；固定 150 步；对五个 split 值计算配对区间。 |
| 强度机制 `tab:maintenance-mechanism` | `tables/metrics_per_split.csv`、`transitions_per_split.csv` | Fixed-FT 和 Fixed-Full-old-a0.1/a1/a10，fixed/150；转移行 `class_id=-1`。 |
| 全 11 类差异 `tab:maintenance-classes` | `tables/classes_per_split.csv`、`class_tests.csv` | Tuned-Full-old−Tuned-FT，selected/150；配对区间及 11 类 Holm 校正。 |
| 计时 `tab:maintenance-cost` | `tables/metrics_summary.csv`、`metrics_per_split.csv` | `first_preparation_s`、`cache_ready_update_s`、`first_use_s`、`service_end_to_end_s`；同主比较筛选。 |
| 全类 PRF `tab:supp-class-prf` | `tables/classes_summary.csv`、`classes_per_split.csv` | Tuned-FT、Tuned-Full-old、Tuned-Random-all 的 precision/recall/F1 分别平均；不可由平均 precision/recall 再算 F1。 |
| 节点预测转移 `tab:supp-node-transitions` | `tables/transitions_per_split.csv` | selected/150、`class_id=-1`；区分“相对于 FT 的分歧”和“相对于 Original 的保护/纠错”。 |
| 目标与选择政策 `tab:supp-target-selection` | `tables/metrics_per_split.csv` | 同强度 Full-input−Full-old；Full-old−Structure；Coverage 比较须用固定协议。 |
| 参考类别组成 `tab:supp-reference-composition` | `tables/reference_composition_per_split.csv` | 按 family、Primary/Secondary、class 分组；同正式分层平均。 |
| 参考度分布 `tab:supp-reference-degree` | 同上 | `mean_degree`、`degree_p25`、`degree_median`、`degree_p75`；它们是组内统计量的平均，不是把参考节点合并后重算的分位数。 |
| 开发搜索成本 `tab:supp-development-cost` | `tables/development_search_costs.csv` | 仅完整重新开发后有数据；计算方法见下一节。 |
| 关系辅助指标 `tab:supp-relation-metrics` | `tables/metrics_summary.csv`、`metrics_per_split.csv` | `task=relation`；AUROC、Brier、NLL；SD 来自五个 split 的样本 SD。 |
| 撤回关系分数响应 | `tables/metrics_per_split.csv`；`formal/requests/relation*/q*/context.npz` 与方法预测 NPZ | `withdrawn_score`、`withdrawn_change`、`withdrawn_post_input_change`；原始均值和绝对变化还需原始数组，见下文。 |
| 三次重训参考变异 | `tables/retrain_variability.csv` | 每个 split 首个 base 的首个请求，合计五个请求；不能将此子集当作全部 20 个关系请求。 |

`scripts/summarize_paper.py` 为本次整理新增的表格摘要工具，不在本包 32 份原执行源码的依赖子集中。它读取运行后生成的表格；导出目录为 `reproduction/paper_tables/`：

- `metrics_split_intervals.csv`、`classes_split_intervals.csv`、`transitions_split_intervals.csv`、`reference_composition_split_intervals.csv`：各 split 表的均值、样本 SD 和区间。
- `paired_contrasts_per_split.csv` / `paired_contrasts_intervals.csv`：正文配对比较及附录目标、结构和覆盖比较；配对差值正负 split 数一并给出。
- `withdrawal_interaction_per_split.csv` / `withdrawal_interaction_intervals.csv`：四条件比较，从指标表重建并核对冻结交互表。
- `protection_correction_per_split.csv` / `protection_correction_intervals.csv`：固定和调优协议的保护收益、纠错差值及净正确数量差。
- `relation_withdrawal_per_request.csv` / `relation_withdrawal_per_split.csv` / `relation_withdrawal_intervals.csv`：撤回关系有符号及绝对响应。
- `retrain_reference_per_split.csv` / `retrain_reference_intervals.csv`、`retrain_reference_pairs_per_split.csv` / `retrain_reference_pairs_intervals.csv`：五请求重训诊断及相对 FT 的配对距离。
- `development_costs.csv`：仅有本轮开发结果时生成；`README.json` 标记所用协议和开发成本是否可用。

撤回关系的原始分数来自 `context.npz` 的 `original_withdrawn` 和 `input_withdrawn`。方法分数来自该方法、该预算与模式的预测 NPZ 的 `withdrawn_probabilities`。有符号响应为 `mean(method − input)`，绝对响应为 `mean(abs(method − input))`，随后按请求、base、split 层级平均。二者不能从同一个均值互相推出；冻结执行代码会保存所需数组。

转成论文显示单位时：质量、AUROC、类别 PRF 乘 100；质量差值及区间乘 100 后单位为百分点；主表 JS 的显示值乘 1000。JS 本身为散度，使用自然对数；NLL 单位为 nats。

## 4. 成本范围与开发成本复算

正式维护计时以运行产物中的字段为准：

- `cache_ready_update_s`：输入处理、当前请求参考设置和优化/验证。
- `first_preparation_s`：首次使用归属该方法的可复用准备成本。
- `first_use_s`：前两项之和。
- `service_end_to_end_s`：first use 再加模型设置和最终推理。
- 开发候选的 `training_and_validation_s`、`wall_before_receipt_s` 是另一个计时范围；后者包括候选级设置和评估，不包含复用 base 准备，也不是多 GPU 搜索的自然流逝总时间。

开发成本表使用每任务、每方法族 2 个 split × 9 个候选。`development_search_costs.csv` 的两项运行时间对 18 行求和；准备成本先在每个开发 base 内取均值，再跨 base 取均值。本协议开发阶段每 split 只有一个 base，因此可按 split 识别该 base。下列代码只读取本轮生成文件，并输出本轮汇总；若跳过 development 则拒绝复算这张表：

```python
from pathlib import Path
import pandas as pd

path = Path("reproduction/tables/development_search_costs.csv")
df = pd.read_csv(path)
required = {"task", "family", "split", "first_preparation_s",
            "training_and_validation_s", "wall_before_receipt_s"}
assert required <= set(df.columns), "先完整运行 development 与 report"
keys = ["task", "family"]
assert (df.groupby(keys).size() == 18).all()
prep = df.groupby(keys + ["split"]).first_preparation_s.mean().groupby(keys).mean()
totals = df.groupby(keys)[["training_and_validation_s", "wall_before_receipt_s"]].sum()
print(totals.join(prep.rename("base_preparation_mean_s")).to_string())
```

本包不填入任何历史计时结果。改变 GPU、并行数、软件环境后需要重新测量成本；复现算法和统计口径不等于保证获得相同秒数。

## 5. 本包的分析范围与校验

在同一解压目录完成训练和正式汇总后，运行本轮相同 protocol 的分析阶段。例如，采用论文固定配置的本轮运行使用：

```bash
python scripts/reproduce.py --stage analysis --protocol paper
```

如果前面使用的是 `redevelop`，此处也须指定 `--protocol redevelop`。该阶段运行新增的 `scripts/summarize_paper.py`，只读取本轮产物生成摘要，不训练或选择模型。报告原始分层表由冻结 `report.py` 生成，它会逐运行读取预测数组重新计算质量指标、核对执行时记录，并检验正式运行和源码身份。

旧工程的独立 `review/` 和 `validation/` 脚本未打入此包。它们包含独立逐对象指标复核、逐节点分歧导出、关系 raw 表交叉核查以及原批次状态检查。这些深层复核不应被描述为已随包提供或已在本轮执行；重新训练后保存的原始数组仍可用于后续独立核查。包内没有依赖历史 `finalization_status.json` 的运行入口，也不需导入旧结果以获得完成状态。

旧 `supplementary_experiments/` 分析器及旧 HASI 基线/消融属于其他实验批次，不是当前论文的复现入口。本包仅保留当前维护实验所需的 32 份执行源码的依赖子集；模块和输出目录已去掉日期，训练/统计算法未改，原始与当前哈希见 EXPERIMENT_SOURCE_MANIFEST.json；其他历史模块与实验入口已裁去。

正式运行失败、预测产物缺失或审计未通过时，应先解决具体错误，不应手工填写完成状态或复制旧表格跳过校验。包内实现测试检查代码行为，不替代真实数据上的完整复现实验；打包时未重新执行论文完整训练。
