# 主文核心图与补充材料完整图

主文 Figure 4 为 2×2 核心机制图；完整机制、类别和成本图分别位于独立补充材料的 Figures S1–S3。`plot_mechanism_core.py` 生成主文核心图，`plot_results.py` 生成完整图。训练、选参和统计入口见包根 [README.md](../README.md)。

## 输入与相对路径

CSV 不随代码包提供。脚本以自身目录为基准读取 `data/<文件名>.csv`，不依赖启动目录，也没有机器固定路径。`publication_figures/data/` 已加入根目录 `.gitignore`。

```text
publication_figures/
  plot_mechanism_core.py
  plot_results.py
  data/
    table09_mechanism.csv
    table10_tradeoff.csv
    tableA1_classwise.csv
    table13_cost.csv
```

| 输入文件 | 主文与补充材料中的用途 |
| --- | --- |
| `data/table09_mechanism.csv` | 主文 Figure 4(a–c)：表示漂移、纠错数、原正确预测损失；补充 Figure S1(a–e)：完整 Fixed 强度扫描。 |
| `data/table10_tradeoff.csv` | 主文 Figure 4(d)：Fixed 默认 a=1 的收益、代价和净正确计数；补充 Figure S1(f–g)：Fixed 与 Tuned 的配对权衡。 |
| `data/tableA1_classwise.csv` | 补充 Figure S2 和 Table S6：完整十一类结果；文件名保留原约定，不表示当前表号。 |
| `data/table13_cost.csv` | 补充 Figure S3 和 Table S9：准备、缓存更新、首次使用和服务成本；文件名保留原约定。 |

这些 CSV 是绘图所用的汇总表，不是原始逐对象预测数组。精确重绘论文核心图时，前两份输入的 SHA-256 必须匹配 `plot_mechanism_core.py` 中记录的论文绘图输入指纹；脚本将输入、输出指纹和面板对应写入 `figure_manifest.json`。该检查不替代训练与逐对象结果复核。新实验产生的汇总表与论文绘图输入表须分别保存和报告。

## 生成命令与输出

在代码包根目录安装 `requirements.txt` 中的绘图依赖后执行：

```bash
# 仅需前两份输入：主文 Figure 4
python publication_figures/plot_mechanism_core.py

# 需要四份输入：补充 Figures S1–S3
python publication_figures/plot_results.py
```

输出保存在 `publication_figures/`：

| 输出 | 论文位置 |
| --- | --- |
| `fig1_mechanism_core.png`、`figure_manifest.json` | 主文 Figure 4 及绘图来源记录。 |
| `fig1_mechanism_tradeoff.*`、对应预览 | 补充 Section S2，Figure S1：完整七面板机制图。 |
| `fig2_classwise_cost.*`、对应预览 | 补充 Section S3，Figure S2：十一类 F1 差异。 |
| `fig3_maintenance_cost.*`、对应预览 | 补充 Section S4，Figure S3：准备与缓存更新成本。 |
| `appendix_table13.tex` | 四种计时范围的 LaTeX 表，内容对应补充 Table S9。 |

完整脚本为三张图各生成 600 dpi PNG、PDF、SVG 和 150 dpi 预览。两脚本直接使用输入表中的 SD、配对区间和 Holm p，不从舍入均值重算区间。主文 Figure 4(a–c) 及完整 Figure S1(a–e) 使用五个 split 均值的 mean ± 1 sample SD；主文 Figure 4(d) 及完整 Figure S1(f–g) 使用配对 95% t 区间，df=4。主文 (d) 对应完整图原 (f)，不能沿用完整图的面板引用。

## 样式与字体

核心图使用白底四面板，减少图内标题和装饰；完整图保留浅青蓝背景、深蓝边框及原面板组织。优先使用 Times New Roman，未安装时按顺序使用 Liberation Serif、STIXGeneral 或 DejaVu Serif；数学符号使用 STIX。字体选择不改变数据或统计口径。
