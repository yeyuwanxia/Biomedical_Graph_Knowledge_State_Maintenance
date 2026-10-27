# 论文实验与代码对应说明

本仓库实现论文中的 Hetionet 实体类型分类（node）和 Compound–treats–Disease 关系预测（relation）实验，入口为 `experiments/knowledge_state_maintenance/`。

本包包含代码、协议、配置和说明，不包含数据或实验结果 CSV。下文实验流程中的 `data/`、模型、预测数组、运行统计 CSV、审计 JSON 和图表由读者准备数据并运行后生成。论文第 5 章为 Experiments，完整诊断位于独立补充材料 S1–S4。主文 Figure 4 的核心脚本读取两份 CSV，补充 Figures S1–S3 的完整脚本读取原四份 CSV，均位于 `publication_figures/data/`；输入说明见 [publication_figures/README.md](../publication_figures/README.md)。完整实验的数据准备见 [DATA_PREPARATION.md](DATA_PREPARATION.md)，执行入口与环境安装以包根 README 为准。

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
6. 执行 `analysis`，由 `scripts/summarize_paper.py` 读取当前运行生成的正式统计表，导出论文主要比较、机制、类别、关系和计时摘要。

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

## 3. 主文、补充材料与方法性附录对应

以下按论文图表编号与 LaTeX 标签对应。主文第 5 章 Experiments 的两个表为 Table 2（设置）和 Table 3（总体结果），核心图为 Figure 4；方法控制 Table 1 与方法性附录 A/B 保留在主文。补充材料的 S 前缀编号独立计算。

配置表来自协议或配置文件；统计表来自运行后生成的 `reproduction/tables/` 和 `reproduction/paper_tables/`。下表来源中的 `tables/`、`paper_tables/` 均以 `reproduction/` 为基准。CSV 文件名中的旧数字不代表当前论文编号。

| 论文位置与标签 | 配置或统计来源 | 筛选/汇总方式 |
| --- | --- | --- |
| 主文 Table 1，比较条件 `tab:study-controls` | `reproduction/PROTOCOL.md`、`run.py` 的 `formal_specs` | Original、Input-only、Continue-FT、Fixed-FT，以及参考目标与 Fixed/Tuned 标签。 |
| 主文 Table 2，数据与任务 `tab:dataset-tasks` | `docs/DATA_PREPARATION.md`、`runtime.py` 的数据构建函数 | 核对节点、消息边、特征、任务划分和撤回；保留两任务预处理差异及去重、自环口径。 |
| 补充 Section S1，Table S1，训练设置 `tab:core-settings` | `configs/paper_protocol.json`、`reproduction/PROTOCOL.md`、`runtime.py` | 模型、优化器、150 步预算、九候选搜索、参考配额与执行设置。 |
| 补充 Section S1，Table S2，选定配置 `tab:selected-configurations` | `configs/paper_protocol.json` 的 `tasks.*.selected_configs` | paper 模式使用记录配置；redevelop 重新选择并记录差异。 |
| 补充 Section S1，Table S3，预设判据 `tab:engineering-criteria` | `reproduction/PROTOCOL.md`、`report.py` 的决策汇总 | 质量差区间下界 >−0.005（node）或 >−0.01（relation），使用 [0,1] 尺度；平均 JS ≤FT 的 0.90 倍且 JS 差区间上界 <0；cached update ≤FT 的 1.25 倍且严格低于 Matched-Retrain；三项须同时满足。 |
| 主文 Table 3，总体结果 `tab:maintenance-main` | `tables/metrics_summary.csv`、`metrics_per_split.csv` | Tuned selected/150，加 Input-only selected/0、Matched-Retrain selected/300；保留全部七方法的质量与 `retrain_js`。 |
| 补充 Section S2，Table S4，配对结果 `tab:maintenance-paired` | `tables/paired_intervals.csv`、`paper_tables/paired_contrasts_intervals.csv` | 逐 split 计算差值和区间；质量差转为百分点，不能相减两个非配对 CI。 |
| 补充 Section S2，Table S5，输入与训练交互 `tab:maintenance-interaction` | `tables/withdrawal_interaction.csv`、`paper_tables/withdrawal_interaction_intervals.csv` | `(Fixed-FT−Input-only)−(Continue-FT−Original)`；固定 150 步，用五个 split 值计算配对区间。 |
| 主文 Figure 4(a–c)，核心强度干预 `fig:mechanism-tradeoff`；补充 Figure S1(a–e)，完整扫描 | `tables/metrics_per_split.csv`、`transitions_per_split.csv`；绘图输入 `table09_mechanism.csv` | Fixed-FT 与 Fixed-Full-old-a0.1/a1/a10，fixed/150；节点转移行 `class_id=-1`；五个 split 均值的 mean ± sample SD。 |
| 主文 Figure 4(d)，Fixed 默认权衡；补充 Figure S1(f–g)，Fixed/Tuned 完整权衡 | `paper_tables/protection_correction_per_split.csv`、`protection_correction_intervals.csv`；绘图输入 `table10_tradeoff.csv` | 主文 (d) 对应完整图原 (f)，仅取 Fixed a=1；完整图另含 Tuned。保护收益−纠错成本=净正确计数差，误差为配对 95% t CI，df=4。 |
| 补充 Section S3，Figure S2，类别图 `fig:classwise-cost`；Table S6，完整十一类 `tab:maintenance-classes` | `tables/classes_per_split.csv`、`class_tests.csv`；绘图输入 `tableA1_classwise.csv` | Tuned-Full-old−Tuned-FT，selected/150；完整十一类 F1、支持量、配对区间和 Holm 校正。图展示十一类并突出四类贡献，十一类差的平均对应 Macro-F1 差。 |
| 补充 Section S3，Table S7，目标与选点 `tab:supp-target-selection` | `paper_tables/paired_contrasts_intervals.csv`、`tables/metrics_per_split.csv` | Fixed 同强度目标、结构及覆盖对照，以及 Tuned Full-old−Structure；分别筛选 fixed/150 和 selected/150。 |
| 补充 Section S3，Table S8，关系辅助指标 `tab:supp-relation-metrics` | `tables/metrics_summary.csv`、`metrics_per_split.csv` | `task=relation`；方法和模式同主比较，含 Input-only 与 Matched-Retrain；AUROC、Brier、NLL，SD 来自五个 split 均值。 |
| 补充 Section S4，Table S9，完整计时 `tab:maintenance-cost`；Figure S3 `fig:maintenance-cost` | `tables/metrics_summary.csv`、`metrics_per_split.csv`；绘图输入 `table13_cost.csv` | `first_preparation_s`、`cache_ready_update_s`、`first_use_s`、`service_end_to_end_s`；方法与输出模式同主比较；首次使用 SD 包含分量协方差。 |
| 主文 Appendix A，参考选择定义 `app:reference-selection`；Table A.1，参考度 `tab:supp-reference-degree` | `runtime.py` 的 `compute_components`、`select_refs`；`tables/reference_composition_per_split.csv` | 保留评分、选择定义及参考组诊断；按政策与 Primary/Secondary 汇总 `mean_degree`，再平均五个 split；Full-input 与 Full-old 成员相同。 |
| 主文 Appendix B，输入与复现约定 `app:current-reproducibility` | `docs/DATA_PREPARATION.md`、`docs/ENVIRONMENT.md`、`runtime.py`、`report.py` | 特征、负例、指标数值约定和实现来源；保留历史环境记录不完整、原 cache/checkpoints/逐对象数组未交付的限制。 |
| 补充 Section S3，撤回关系分数响应 | `tables/metrics_per_split.csv`；`formal/requests/relation*/q*/context.npz` 与方法预测 NPZ | 有符号变化和绝对变化分别计算，再按请求、base、split 汇总；撤回关系仍是已知正例。 |
| 补充 Section S3，关系重训变异 | `tables/retrain_variability.csv`、`paper_tables/retrain_reference_intervals.csv` | 每 split 首个 base 的首个请求，合计五个请求、每请求三次重训；区别于主表全部二十个关系请求。 |

主文 Figure 4 由 `publication_figures/plot_mechanism_core.py` 生成；补充 Figures S1–S3 由保留的 `publication_figures/plot_results.py` 生成。输入、输出和指纹规则见绘图目录 README；两脚本使用汇总表绘图，不重新训练或重算配对区间。

`scripts/summarize_paper.py` 是独立的表格摘要工具，不在 32 份实验来源文件的依赖子集中。它读取运行后生成的表格；导出目录为 `reproduction/paper_tables/`：

- `metrics_split_intervals.csv`、`classes_split_intervals.csv`、`transitions_split_intervals.csv`、`reference_composition_split_intervals.csv`：各 split 表的均值、样本 SD 和区间。
- `paired_contrasts_per_split.csv` / `paired_contrasts_intervals.csv`：正文配对比较及目标、结构和覆盖比较；配对差值正负 split 数一并给出。
- `withdrawal_interaction_per_split.csv` / `withdrawal_interaction_intervals.csv`：四条件比较，从指标表重建并核对冻结交互表。
- `protection_correction_per_split.csv` / `protection_correction_intervals.csv`：固定和调优协议的保护收益、纠错差值及净正确数量差。
- `relation_withdrawal_per_request.csv` / `relation_withdrawal_per_split.csv` / `relation_withdrawal_intervals.csv`：撤回关系有符号及绝对响应。
- `retrain_reference_per_split.csv` / `retrain_reference_intervals.csv`、`retrain_reference_pairs_per_split.csv` / `retrain_reference_pairs_intervals.csv`：五请求重训诊断及相对 FT 的配对距离。
- `development_costs.csv`：仅有当前运行的开发结果时生成；`README.json` 标记所用协议和开发成本是否可用。

撤回关系的原始分数来自 `context.npz` 的 `original_withdrawn` 和 `input_withdrawn`。方法分数来自该方法、该预算与模式的预测 NPZ 的 `withdrawn_probabilities`。有符号响应为 `mean(method − input)`，绝对响应为 `mean(abs(method − input))`，随后按请求、base、split 层级平均。二者不能从同一个均值互相推出；冻结执行代码会保存所需数组。

转成论文显示单位时：质量、AUROC、类别 PRF 乘 100；质量差值及区间乘 100 后单位为百分点；主表 JS 的显示值乘 1000。JS 本身为散度，使用自然对数；NLL 单位为 nats。

## 4. 成本范围与开发成本复算

正式维护计时以运行产物中的字段为准：

- `cache_ready_update_s`：输入处理、当前请求参考设置和优化/验证。
- `first_preparation_s`：首次使用归属该方法的可复用准备成本。
- `first_use_s`：前两项之和。
- `service_end_to_end_s`：first use 再加模型设置和最终推理。
- 开发候选的 `training_and_validation_s`、`wall_before_receipt_s` 是另一个计时范围；后者包括候选级设置和评估，不包含复用 base 准备，也不是多 GPU 搜索的自然流逝总时间。

开发成本表使用每任务、每方法族 2 个 split × 9 个候选。`development_search_costs.csv` 的两项运行时间对 18 行求和；准备成本先在每个开发 base 内取均值，再跨 base 取均值。本协议开发阶段每 split 只有一个 base，因此可按 split 识别该 base。下列代码只读取当前运行生成的文件，并输出当前运行的汇总；若跳过 development 则拒绝复算这张表：

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

在同一解压目录完成训练和正式汇总后，按训练时相同的 protocol 执行分析阶段。例如，采用论文固定配置的运行使用：

```bash
python scripts/reproduce.py --stage analysis --protocol paper
```

如果前面使用的是 `redevelop`，此处也须指定 `--protocol redevelop`。该阶段运行 `scripts/summarize_paper.py`，只读取当前运行产物生成摘要，不训练或选择模型。报告原始分层表由冻结 `report.py` 生成，它会逐运行读取预测数组重新计算质量指标、核对执行时记录，并检验正式运行和源码身份。

本包提供实验执行、分层汇总和表格整理工具；独立逐对象复核与原运行状态检查工具未包含。新运行保存的预测数组可用于进一步独立核查。

维护实验的来源清单包含 31 份 Python 文件和 1 份协议。`EXPERIMENT_SOURCE_MANIFEST.json` 记录原始与分发路径、哈希、模块重命名及协议序言编辑；训练与统计算法保持一致。

正式运行失败、预测产物缺失或审计未通过时，应先解决具体错误，不应手工填写完成状态或复制旧表格跳过校验。包内实现测试检查代码行为，不替代真实数据上的完整复现实验；尚未通过重新执行完整论文训练验证数值一致性。
