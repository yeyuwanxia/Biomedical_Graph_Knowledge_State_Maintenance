# S2 / S3：历史证据与待补信息

**S2 仍部分缺失；S3 的训练实现已有证据，Adam 其余历史参数仍待补。** 代码完整性检查或合成测试通过，不能证明这些历史信息已补齐。本页只整理可追溯证据，没有采集当前机器环境，没有修改论文或补造历史值。

| 检查项 | 当前状态 | 包内依据或仍缺内容 |
| --- | --- | --- |
| S2 核心软件、线程设置 | 已有历史记录 | Python 3.12.7、PyTorch 2.9.0+cu128、CUDA runtime 12.8、PyG 2.8.0 等；可见设备 0,1,2、每 worker 4 线程。完整摘录见 [HISTORICAL_ENVIRONMENT.json](HISTORICAL_ENVIRONMENT.json)。 |
| S2 GPU 型号、并发方式 | 有执行协议 | [PROTOCOL.md](../reproduction/PROTOCOL.md) 第 51 行写明三张可见 T4、每卡一个 worker。这不是每卡显存实测记录。 |
| S2 历史硬件与系统 | **待补** | 当时 CPU 型号、主机 RAM、每卡实际显存、OS 发行版/版本、GPU 驱动。CUDA runtime 12.8 不能代替驱动版本。 |
| S2 完整依赖锁 | **待补** | 当时完整的 pip/conda 环境导出，或能够核验身份的原环境镜像；当前 requirements 只固定已知核心版本。 |
| S3 full-graph / full-batch | 已有源码证据 | 两任务共用 [runtime.py](../experiments/knowledge_state_maintenance/runtime.py) 的 fit，第 282–300 行；每轮对完整消息图和全部保留训练对象计算一次损失、backward、step。 |
| S3 训练 float32 | 已有源码及历史工件核验 | 节点预处理 hetionet.py 第 449 行、关系特征 runtime.py 第 177–178 行；已有审计核对 10 个正式输入和 30 个基础模型，历史指纹匹配且浮点特征/参数为 float32。包内仅附核验摘要，不含张量或模型。 |
| S3 AMP、梯度缩放、scheduler | 已有源码证据 | 共用训练循环直接反向传播和 Adam 更新，没有 autocast、GradScaler、学习率 scheduler；单次 fit 内不改变学习率。 |
| S3 Adam 其余参数 | **历史依据待补** | 冻结代码仅显式传入 lr 和 weight_decay；betas、eps、amsgrad 等未显式传入，历史记录未保存其有效值。 |

float32 指训练特征与模型参数。`runtime.py` 第 122–123 行在评价概率时使用 `logits.double()`；不能扩大为“整个流程的全部运算均为 float32”。参考评分中的分批计算也不改变 GCN 训练为 full-batch 的结论。

## Adam 参数为什么仍缺依据

两任务的准确调用均为：

```python
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.0005)
```

出处为包内 `experiments/knowledge_state_maintenance/runtime.py:292`。`state(model)` 保存的是模型参数；`run.py:57,204–207` 保存基础模型、选中模型及模型快照，未保存 `optimizer.state_dict()`、`param_groups` 或 `defaults`。训练返回信息也没有这些选项。

因此当前可确认的是“其余选项继承当时安装的 Adam 实现”，无法把 `betas=(0.9,0.999)`、`eps=1e-8`、`amsgrad=False` 写成从本轮历史日志核实的值。这里没有判定这些数值错误；缺的是证明其在当时生效的材料。foreach、fused 等其他隐式选项同样应由准确安装实现或有效参数记录确认。

## 什么材料可以补齐

- **S2：** 当时运行机器的硬件/系统/驱动记录，以及完整环境导出或原环境备份；需要能对应到这轮正式实验。其他批次的记录不能直接替代。
- **S3 Adam：** 当时打印的有效参数、包含 optimizer 状态的 checkpoint，或能够证明当时确实使用的 PyTorch 安装包/源码身份及其构造调用链。若依据原安装包事后恢复，须注明恢复方法。

只提供“常见默认值”不满足历史核验要求。现在运行硬件查询、导出 pip freeze、给代码补显式参数，都不能单独证明过去的配置；重新训练产生的也应作为新一轮记录。

若这些材料没有保留，当前就应保留 S2 和 S3 Adam 的待补说明，不将其标为已完成。本包没有修改原论文的 S2/S3 标记。

## 已附证据的范围

[HISTORICAL_ENVIRONMENT.json](HISTORICAL_ENVIRONMENT.json) 仅摘录本轮历史环境字段、版本、执行协议和既有 dtype 核验摘要，并记录各原始记录的文件名、SHA256 与提取字段。未知值明确为 null。没有复制测试输出、预测指标、数据集、模型或其他实验结果；没有复制历史核验工具所用的当前解释器信息。

新运行时，`scripts/reproduce.py` 会生成 `reproduction/reproduction_environment.json`，记录该次运行的软件、硬件查询和新环境中 Adam 的默认选项。这是新复现的证据，不用于回填本页的历史缺项。
