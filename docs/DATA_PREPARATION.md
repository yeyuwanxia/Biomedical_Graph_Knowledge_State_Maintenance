# 数据准备（压缩包不含数据集）

本包只提供代码和说明。原始图、处理缓存、任务划分、模型参数及实验结果均需在解压后由使用者获取或重新生成。下列规模、哈希和协议用于确认输入，不是随包附送的实验结果。

## 1. 获取原始 Hetionet v1.0

从 [Hetionet 官方 JSON 目录](https://github.com/hetio/hetionet/tree/main/hetnet/json) 获取 `hetionet-v1.0.json.bz2`。该文件是含 `nodes`、`edges` 等对象的 bzip2 压缩 JSON；不要下载只有图模式的 `hetionet-v1.0-metagraph.json`。目录及文件名称已于 2026-09-23 核对。

冻结代码记录的下载地址为：

```text
https://github.com/hetio/hetionet/raw/main/hetnet/json/hetionet-v1.0.json.bz2
```

下载由使用者自行完成；`scripts/prepare_data.py` 不联网。请保留原始 `.bz2` 字节，不要解压后重新压缩。论文正式关系任务所记录的原始文件 SHA256 为：

```text
a342ab57e9073e6c02bb5e109d1f16917e6f933be4e5c77ebbaebfa26b984c19
```

包装脚本严格校验此哈希；不匹配时停止，不会悄悄换用其他版本。官方分支 URL 不是本包的数据镜像，文件内容以哈希为准。数据使用条件请参见官方仓库及原始数据来源。

## 2. 本地建立节点缓存

在代码包根目录、按根目录说明装好 Python 依赖后执行：

```bash
python scripts/prepare_data.py --raw-file ../datasets/hetionet-v1.0.json.bz2
```

执行时才会生成：

```text
data/
  raw/Hetionet/hetionet-v1.0.json.bz2
  raw/Hetionet/processed_full_nosource/data.pt
  raw/Hetionet/processed_full_nosource/metadata.json
  preparation_receipt.json
```

也可以先自行把原文件放到上述目标位置，再执行 `python scripts/prepare_data.py`。重复执行会核验并复用完整的已有缓存。遇到残缺或协议不同的缓存，脚本停止；建议重新解压一份干净代码包建立新的复现目录，避免混用已有实验产物。脚本不会重写不同的原图文件，也不自动启动训练。

仅核查既有输入和缓存：

```bash
python scripts/prepare_data.py --check-only
```

此命令仍需载入缓存及 PyTorch/PyG 等依赖，但不会写文件或计算新的图特征。首次建立全图缓存会解压约 225 万条边，并由 NetworkX 计算三角形、核心数和 PageRank，需要较多 CPU 时间和内存。本代码包制作时只验证了微型合成图，未重新执行完整图预处理或完整训练。

## 3. 节点任务与论文一致的构图规则

包装脚本直接调用冻结的 `src/data/hetionet.py::load_hetionet_full_nosource`，没有另写一套特征实现。它使用全部 11 种实体类型，不按类别截断或抽样；沿用原 JSON `metanode_kinds` 的类别顺序以及各类别内原始节点顺序。节点类型为分类标签，不作为输入特征。

| 项目 | 本包使用的协议或检查值 |
| --- | --- |
| 原图 | 47,031 个节点，2,250,197 条原始关系记录 |
| 节点类型 | 11 类，`min_nodes_per_class=1`，无每类节点上限 |
| 处理边 | 去自环，添加反向边，再按端点对去重 |
| 最终 `edge_index` | 4,215,418 个有向条目，即 2,107,709 个无向节点对 |
| 孤立节点 | 保留 1,873 个，最终仍有 47,031 个节点 |
| 特征 | 20 维结构特征，`float32` |
| 标签顺序 | Anatomy、Biological Process、Cellular Component、Compound、Disease、Gene、Molecular Function、Pathway、Pharmacologic Class、Side Effect、Symptom |
| 划分 | 此步不建立训练/验证/测试划分；后续冻结实验入口按 split seed 分层划分 |

结构特征按以下顺序计算，完整数学实现以冻结源码为准：

1. `log_degree`、`sqrt_degree`、`normalized_degree`。
2. `log_raw_in_degree`、`log_raw_out_degree`、`raw_in_share`、`raw_out_share`。
3. `log_neighbor_degree_mean`、`normalized_neighbor_degree_mean`、`normalized_neighbor_degree_std`、`normalized_neighbor_degree_min`、`normalized_neighbor_degree_max`、`neighbor_degree_ratio`。
4. `local_clustering_coefficient`、`log_triangle_count`、`normalized_triangle_count`。
5. `normalized_core_number`、`log_core_number`、`pagerank`、`log_scaled_pagerank`。

这里 `raw_in_degree` 和 `raw_out_degree` 在去除自环后、双向化和去重之前累计，因此保留原始关系记录的方向和重复计数；其余图结构统计基于同质、无向简单图。孤立点也会加入图中。PageRank 参数为 `alpha=0.85, max_iter=100, tol=1e-6`。特征不包含实体类型 one-hot、来源数据库指示或关系类型计数。该节点任务固定原始全图结构特征，不能据此声称删除请求后已删除一切原始数据的派生影响。

## 4. 关系任务输入由后续入口生成

`prepare_data.py` 为关系任务准备的是同一个原始 `.bz2` 文件；它不提前生成关系任务缓存。后续 `scripts/reproduce.py` 的 prepare 阶段会调用冻结 `experiments/knowledge_state_maintenance/runtime.py`：

- 按 `(kind, str(identifier))` 排序节点。关系任务的节点顺序与节点分类缓存的顺序不能互换。
- 以 755 条唯一 Compound–Disease `treats` 对作为正例；共有 212,624 个候选化合物–疾病对。
- 输入特征为 11 维节点类型 one-hot 加 32 维高斯身份特征，随机种子 180001，总计 43 维。不要将节点任务的 20 维结构特征用于此任务。
- 从背景图排除所有直接 Compound–Disease 记录，再加入当前划分允许使用的训练 `treats`。构造传播边时去自环、双向化和去重。
- 正例划分为 604/75/76；验证/测试各有 750/760 个未知候选作为负例。训练负例由后续基模型种子选取。

划分及关系缓存写在 `reproduction/data/`，均为本次执行的新产物，未放入压缩包。

## 5. 输入哈希与重新复现的边界

正式节点实验使用的旧 `processed_full_nosource/data.pt` 文件字节 SHA256 为：

```text
7953370be1c45eed015d9683eaa34ce411de70366e229a653dd10e648db0a0a1
```

这里只保留该字符串用于溯源，没有附带旧缓存。重新构建时即使张量相同，PyTorch 序列化格式、库版本或对象元信息仍可能造成文件字节哈希不同；不能保证新缓存重现这个旧哈希。NetworkX/SciPy 等版本也可能导致浮点特征差异。

因此，包装脚本强制校验原始 `.bz2` 哈希、图规模、标签分布、20 维特征模式、去自环/双向/去重及保留孤立节点等协议，并记录新缓存字节哈希、张量内容哈希和依赖版本到 `data/preparation_receipt.json`。张量内容哈希用于同环境的新运行之间比较，不代表已经与作者未随包分发的旧张量逐位验证。已有缓存的核验不会重新计算全部结构特征。

冻结运行入口按本次生成的输入记录新的哈希和完成回执，没有要求新缓存必须等于旧字节哈希；只能在新的复现目录中运行，不能把新缓存与旧任务划分、模型或结果拼接后宣称原始归档重现。论文数值的一致性需等完整实验运行后另行比较。
