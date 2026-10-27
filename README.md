# Biomedical Graph Knowledge-State Maintenance

生物医学图知识状态维护：节点分类、关系预测与表示约束机制实验代码。

对应论文：
**Knowledge-State Maintenance in Biomedical Graph Learning: An Empirical Study of Representation Constraints**。

本仓库提供 Hetionet 实体类型分类、Compound–treats–Disease 关系预测，以及 FT / Matched-Retrain / 表示约束方法的机制与调优比较代码。**不包含数据集、模型权重、训练日志、预测数组或实验结果 CSV。** 实验和绘图输入由使用者准备，运行结果在本地生成。

## 内容与版本

- `experiments/knowledge_state_maintenance/`：论文对应的冻结训练、选参、正式运行和汇总代码。
- `src/`：实验代码所需的共用实现；当前模型定义于 `runtime.Model`。
- `reproduction/PROTOCOL.md`：执行前协议；运行产物由相关命令生成。
- `configs/paper_protocol.json`：开发候选、论文采用的超参数、随机种子与复现实验规模；不含开发验证分数、正式指标或预测。
- `scripts/`：数据准备、可变GPU数量调度、统计整理和包完整性检查工具。
- `publication_figures/`：主文 Figure 4 与补充材料 Figures S1–S3 的排版脚本和输入说明；CSV 数据不随包提供。
- `tests/`：8项实现测试及复现工具的小型合成测试。
- `docs/`：安装环境、数据准备和论文实验与代码对应说明。

论文实验来源清单包含 **32份文件（31份Python及1份协议）**。`EXPERIMENT_SOURCE_MANIFEST.json` 记录原始与分发路径、哈希、模块重命名和协议序言编辑，训练与统计算法保持一致；数据准备、调度和统计摘要等辅助工具单独提供。

## 1. 校验代码包

解压后进入本目录，以下命令仅需Python标准库，不读取数据或调用GPU：

```bash
python scripts/verify_package.py --strict-code-only
python scripts/reproduce.py --protocol redevelop --devices cuda:0 --dry-run
```

`--strict-code-only`用于刚解压、尚未准备数据的目录。生成输入或实验产物后，可用不带该选项的命令核对交付源文件。每个已交付文件的校验值见 `PACKAGE_MANIFEST.json` 和 `SHA256SUMS`。

## 2. 安装与小图测试

详细说明见 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)。建议单独建立环境：

```bash
conda create -n hasi-kbs-repro python=3.12.7 pip
conda activate hasi-kbs-repro
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests -q
```

已记录的核心软件版本被固定；NetworkX、pytest及传递依赖缺少完整历史版本记录，因此本包不声称提供完整历史环境锁。合成测试可以在CPU上执行，不需要Hetionet；完整实验需要CUDA。

## 3. 自行提供并准备数据

获取与论文相同的 Hetionet v1.0 压缩JSON，数据入口、原始SHA256和预处理口径见 [docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md)。本包不自动下载数据，也不包含处理后的缓存。

```bash
python scripts/prepare_data.py --raw-file ../datasets/hetionet-v1.0.json.bz2
python scripts/prepare_data.py --check-only
```

脚本验证原图指纹，调用冻结预处理代码生成20维节点结构特征、节点标签及去重双向图，并保留孤立节点。随后训练入口按各split创建节点和关系任务的输入。首次全图预处理需要时间及主机内存；脚本不会启动训练。

原 `data.pt` 的字节哈希仅供来源对照；新环境的Torch序列化字节可能不同。脚本记录新缓存哈希并检查图构造事实；这不能保证未完整冻结的数值依赖下所有特征逐位一致。

## 4. 两种复现方式

### A. 重做完整开发与正式流程（默认）

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python scripts/reproduce.py --stage all --protocol redevelop
```

顺序为 GPU全图零权重检查 → 开发冻结 → 两个开发split的候选搜索 → 按原规则选参与规划base数 → 五个正式split → 分层汇总 → 论文统计整理。

这种方式可以重新测量开发搜索成本。开发观测会重新计算，因此硬件或数值差异可能使选参/base数不同于论文；差异记录在当前运行生成的 `published_configuration_comparison.json` 中，不强行改成论文值。

### B. 直接按论文已选参数运行正式实验

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python scripts/reproduce.py --stage all --protocol paper
```

沿用 `configs/paper_protocol.json` 中论文已确定的配置，运行节点20个base/40个请求、关系10个base/20个请求及规定参考重复/额外重训。会进行GPU检查、生成当前运行输入和新的 manifest，但不会重新执行开发选参或样本量估计。

**paper方式不重新测量开发搜索成本。** 运行后报告会注明该限制，空的开发成本表不代表成本为零。不要混用两种方式的部分产物；比较两种方式应分别解压到独立目录。

### 单卡与分阶段运行

调度器支持一张或多张CUDA卡；冻结训练逻辑、随机流、loss与选模逻辑不变。设备参数是 `CUDA_VISIBLE_DEVICES` 映射后的逻辑编号，必须包含 `cuda:0` 供原GPU检查使用。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py --stage all --protocol paper --devices cuda:0
```

也可分别执行 `--stage prepare`、`development`、`formal`、`report`、`analysis`；每一步保持相同 `--protocol`。`prepare`包括GPU检查，不等于纯CPU的数据预处理；paper方式跳过development。支持在相同源码、相同输入、运行环境和GPU调度模式下重试，已校验的运行会复用；不覆盖另一种模式或其他输入的运行。

每个base worker默认超时7200秒，可显式用 `--timeout 14400` 等调整，单位为秒。GPU数量、型号或负载改变会影响计时，应将新测耗时作为当前运行环境的结果。

## 5. 新运行会生成什么

所有下列内容在压缩包中均不存在；只有用户执行相关步骤才生成：

| 路径 | 用途 |
|---|---|
| `data/raw/Hetionet/` | 用户提供的原图及生成的节点缓存 |
| `results/EXCLUDED_RESULTS.json` | 新工作区的空历史结果排除声明；仅为原freeze函数的接口文件 |
| `reproduction/reproduction_context.json` | 当前运行模式、源码和输入指纹，防止混合续跑 |
| `reproduction/reproduction_environment.json` | 当前运行机器、包版本、pip freeze和实际Adam默认参数记录 |
| `reproduction/development/`、`formal/` | 当前运行训练、模型、请求、预测及日志 |
| `reproduction/tables/` | 逐请求/逐base/逐split、配对区间、类别和成本表 |
| `reproduction/figures/` | 当前运行生成的质量/距离/成本曲线 |
| `reproduction/paper_tables/` | 进一步整理的分层区间、保护/纠错、关系变化等论文统计 |

源码采用包内相对路径。不要把旧工作区的 `development_freeze.json`、`formal_release.json`、模型或结果混入当前运行。历史原始 manifest 含开发观测，不随代码包分发；当前运行的 manifest 由执行命令重新建立。

论文各部分的实现、表格来源与统计口径见 [docs/PAPER_TO_CODE.md](docs/PAPER_TO_CODE.md)。

## 6. 复绘主文核心图与补充材料完整图

自行准备绘图输入 CSV，放在 `publication_figures/data/`；文件名、统计口径和编号见 [publication_figures/README.md](publication_figures/README.md)。CSV 不随代码包提供，该数据目录已加入 `.gitignore`。在代码包根目录执行：

```bash
# 主文 Figure 4：读取机制与权衡两份 CSV
python publication_figures/plot_mechanism_core.py

# 补充材料 Figures S1–S3：读取原四份 CSV
python publication_figures/plot_results.py
```

核心图脚本生成 `publication_figures/fig1_mechanism_core.png` 和 `figure_manifest.json`；完整脚本保留原机制、类别和成本图的 PNG、PDF、SVG、预览及计时 LaTeX 表输出。两脚本均按自身位置读取相对路径，不启动训练或重新估计区间。

核心图精确重绘要求两份输入的 SHA-256 与脚本记录的论文绘图输入指纹匹配。这些输入是已汇总的绘图表，不是原始逐对象预测数组；指纹匹配只核对绘图来源。重新运行实验得到的结果应单独报告，不能据此声称重现了原预测数组。正式实验结果由前述训练、report 和 analysis 阶段重新计算。

## 验证范围

校验工具检查文件指纹与 Python 语法；`tests/` 提供合成小图及复现工具测试。本包未重新执行完整图预处理或论文训练，数值一致性需在准备数据并完成实验后核对。

所有安装/运行示例均以解压目录为基准使用相对路径；`../datasets/`为用户自行存放输入的示例位置，不是包内数据。源码以`__file__`自动定位项目目录，不含固定机器路径。
