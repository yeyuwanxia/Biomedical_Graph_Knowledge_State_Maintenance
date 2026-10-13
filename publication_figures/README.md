# 论文图4–6绘图

`plot_results.py` 按当前论文图4–6的样式排版。绘图数据不随代码包提供，也不上传 GitHub；请在本地准备四份 CSV。完整实验的训练和统计入口见代码包根目录 README。

## 相对路径

脚本以自身所在目录为基准读取 `data/<文件名>.csv`，不依赖启动命令时的工作目录，也没有固定的桌面或机器路径。将输入放在以下本地位置：

```text
publication_figures/
  plot_results.py
  data/
    table09_mechanism.csv
    table10_tradeoff.csv
    tableA1_classwise.csv
    table13_cost.csv
```

`publication_figures/data/` 已加入根目录 `.gitignore`。

| 输入文件 | 对应内容 |
| --- | --- |
| `data/table09_mechanism.csv` | 图4(a–e)：固定强度机制比较 |
| `data/table10_tradeoff.csv` | 图4(f–g)：保护收益与纠错成本 |
| `data/tableA1_classwise.csv` | 图5：十一类实体 F1 差异 |
| `data/table13_cost.csv` | 图6：准备与缓存更新成本 |

在代码包根目录安装 requirements.txt 中的绘图依赖后执行：

```bash
python publication_figures/plot_results.py
```

输出保存在脚本目录：三张图各自的 600 dpi PNG、PDF、SVG、150 dpi 预览，以及四种计时范围的 LaTeX 表格 `appendix_table13.tex`。绘图使用输入表中已报告的 SD、配对区间和 Holm p，不根据舍入后的均值重算区间。脚本内的数据校验针对论文显示数值；自行准备的表格应使用对应字段、统计口径和单位。

## 图形样式与字体

图4–6与论文方法图采用一致的浅青蓝背景、深蓝边框、蓝绿配色和粗体衬线标题。优先使用 Times New Roman；系统未安装时按顺序使用 Liberation Serif、STIXGeneral 或 DejaVu Serif。数学符号使用 STIX。要得到与当前论文完全相同的字体，请在绘图环境安装 Times New Roman。数据读取和统计口径不受字体选择影响。
