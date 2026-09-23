# Biomedical Graph Knowledge-State Maintenance

生物医学图知识状态维护：节点分类、关系预测与表示约束机制实验代码。

对应 `HASI_KBS_20260920_FULL.zip` / `HASI_KBS_20260920.pdf`：
**Knowledge-State Maintenance in Biomedical Graph Learning: An Empirical Study of Representation Constraints**。

本包只复现这版论文的 Hetionet 节点分类、Compound–treats–Disease 关系预测，以及 FT / Matched-Retrain / 表示约束方法的机制与调优比较。**不包含最初 HASI 与旧基线的实验入口、旧模块消融配置、数据集、模型权重、训练日志、预测数组或实验结果。** 输入与结果文件均由用户运行后在本地生成。

## 内容与版本

- `experiments/knowledge_state_maintenance/`：论文对应的冻结训练、选参、正式运行和汇总代码。
- `src/`：上述代码的依赖闭包，按用户要求同步模块/输出路径命名。当前模型定义于 `runtime.Model`，不使用旧 `src/models` 或旧基线目录。
- `reproduction/PROTOCOL.md`：原执行前协议，本目录打包时仅有该说明文件。
- `configs/paper_protocol.json`：开发候选、论文采用的超参数、随机种子与复现实验规模；不含开发验证分数、正式指标或预测。
- `scripts/`：新增的数据准备、可变GPU数量调度、统计整理和包完整性检查工具。
- `tests/`：原8项实现测试，以及新增工具的小型合成测试。
- `docs/`：安装环境、数据准备和论文实验与代码对应说明。

历史快照共44份文件；本包选取实际依赖的 **32份文件（31份Python及1份协议）**，排除旧 `src/baselines/`、`src/models/`、`src/utils/` 的12份无依赖文件。`EXPERIMENT_SOURCE_MANIFEST.json`记录原始路径/哈希、改名后路径/哈希以及改名范围；改名涉及模块引用，训练与统计算法未改。新增包装脚本与说明另行标识。`src`中仍有通过包初始化间接导入的历史共用实现，保留这些依赖不表示本文重新比较旧方法。

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

这种方式可以重新测量开发搜索成本。开发观测会重新计算，因此硬件或数值差异可能使选参/base数不同于论文；差异记录在本次生成的 `published_configuration_comparison.json` 中，不强行改成论文值。

### B. 直接按论文已选参数运行正式实验

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python scripts/reproduce.py --stage all --protocol paper
```

沿用 `configs/paper_protocol.json` 中论文已确定的配置，运行节点20个base/40个请求、关系10个base/20个请求及规定参考重复/额外重训。会进行GPU检查、生成本次输入和新manifest，但不会重新执行开发选参或样本量估计。

**paper方式不能重新产生附录的开发搜索成本。** 运行后报告会注明该限制，空的开发成本表不代表成本为零。不要混用两种方式的部分产物；比较两种方式应分别解压到独立目录。

### 单卡与分阶段运行

新增调度器支持一张或多张CUDA卡；冻结训练逻辑、随机流、loss与选模逻辑不变。设备参数是 `CUDA_VISIBLE_DEVICES` 映射后的逻辑编号，必须包含 `cuda:0` 供原GPU检查使用。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py --stage all --protocol paper --devices cuda:0
```

也可分别执行 `--stage prepare`、`development`、`formal`、`report`、`analysis`；每一步保持相同 `--protocol`。`prepare`包括GPU检查，不等于纯CPU的数据预处理；paper方式跳过development。支持在相同源码、相同输入、运行环境和GPU调度模式下重试，已校验的运行会复用；不覆盖另一种模式或其他输入的运行。

每个base worker默认超时7200秒，可显式用 `--timeout 14400` 等调整，单位为秒。GPU数量、型号或负载改变会影响计时，应将新测耗时作为本次环境的结果。

## 5. 新运行会生成什么

所有下列内容在压缩包中均不存在；只有用户执行相关步骤才生成：

| 路径 | 用途 |
|---|---|
| `data/raw/Hetionet/` | 用户提供的原图及生成的节点缓存 |
| `results/EXCLUDED_RESULTS.json` | 新工作区的空历史结果排除声明；仅为原freeze函数的接口文件 |
| `reproduction/reproduction_context.json` | 本次模式、源码和输入指纹，防止混合续跑 |
| `reproduction/reproduction_environment.json` | 本次机器、包版本、pip freeze和实际Adam默认参数记录 |
| `reproduction/development/`、`formal/` | 本次训练、模型、请求、预测及日志 |
| `reproduction/tables/` | 逐请求/逐base/逐split、配对区间、类别和成本表 |
| `reproduction/figures/` | 本次生成的质量/距离/成本曲线 |
| `reproduction/paper_tables/` | 进一步整理的分层区间、保护/纠错、关系变化等论文统计 |

源码采用包内相对路径。不要把旧工作区的 `development_freeze.json`、`formal_release.json`、模型或结果复制回来假装本次生成。历史原始manifest含开发观测，因此没有原样打包；本次manifest在用户执行时重新建立。

论文每部分的对应方法、表格来源与统计口径见 [docs/PAPER_TO_CODE.md](docs/PAPER_TO_CODE.md)。约束强度/目标/选点对照属于当前论文的机制实验，不是最初HASI模块消融的旧实验。

## 验证范围

本次打包检查代码指纹、依赖导入、合成小图测试、启动计划与统计工具；没有下载/预处理完整图或重跑论文训练。输入缺失时需要用户自行准备，数值一致性仍受依赖版本与设备影响。本包不声明已复现全文数值，也没有附上旧结果来替代运行。

所有安装/运行示例均以解压目录为基准使用相对路径；`../datasets/`为用户自行存放输入的示例位置，不是包内数据。源码以`__file__`自动定位项目目录，没有打包机器的固定绝对路径。
