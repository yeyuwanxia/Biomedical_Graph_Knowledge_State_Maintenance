# 安装与运行环境

下表列出论文实验记录的核心软件版本，提取自 `development_freeze.json` 和 `validation/environment_and_tests.json`；本包不附带这两个原运行记录文件。

| 项目 | 论文实验记录的配置 |
| --- | --- |
| Python | 3.12.7，Anaconda 构建 |
| PyTorch | `2.9.0+cu128` |
| PyTorch 使用的 CUDA runtime | 12.8 |
| torch-geometric | 2.8.0 |
| NumPy | 1.26.4 |
| pandas | 2.2.2 |
| SciPy | 1.16.3 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.10.6，仅用于图形输出 |
| 可见 GPU | `CUDA_VISIBLE_DEVICES=0,1,2` |
| CPU 线程 | 每个 worker 4 线程 |

执行协议采用三张 T4、每张卡一个 worker。完整历史硬件、系统、驱动和依赖锁记录未全部保留；Adam 其余有效选项未单独记录。本仓库不包含原实验节点缓存、模型检查点和逐对象预测数组。

## 建立新环境

在解压后的代码包根目录执行。下面使用 conda 建立 Python 3.12.7 环境；已有相同 Python 的用户也可自行建立 venv。

```bash
conda create -n hasi-kbs-repro python=3.12.7 pip
conda activate hasi-kbs-repro
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
```

PyTorch 2.9.0 与 CUDA 12.8 的安装命令见 [PyTorch 官方历史版本安装页](https://pytorch.org/get-started/previous-versions/)。本项目不需要安装 torchvision 或 torchaudio。实际机器仍需具有兼容驱动，是否可用应以以下检查为准。

`requirements.txt` 锁定了论文实验记录的七个 Python 包；`networkx` 和 `pytest` 是源码和测试需要、但历史精确版本未记录的依赖，故保持未锁定。传递依赖也未全部锁定。这不是完整的历史环境锁文件，应在新环境安装后保存实际版本。

```bash
python -c "import torch, torch_geometric; print('torch:', torch.__version__); print('PyG:', torch_geometric.__version__); print('CUDA runtime:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('visible GPU count:', torch.cuda.device_count())"
python -m pytest tests/test_knowledge_state_maintenance.py -q
```

这组单元测试使用合成小图，可以在没有 Hetionet 数据的情况下运行；它检查零权重与 FT 一致性、验证选模、撤回实现、关系泄漏审计、参考配额、开发候选数和分层汇总。测试通过不等同于完整论文实验已经复现。

## GPU 数量与计时

冻结原入口 `python -m experiments.knowledge_state_maintenance.run run` 固定要求三张可见 GPU。`scripts/reproduce.py` 接受 `--devices cuda:0`，可在一张可见 GPU 上依次执行全部任务；默认配置使用三张可见 GPU。设备编号是 `CUDA_VISIBLE_DEVICES` 筛选后的逻辑编号。

单卡示例（数据准备和模式选择见根目录 README）：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py --stage all --devices cuda:0
```

原训练方法、随机种子和评价逻辑保留在冻结源码中；复现入口负责准备步骤和任务调度。改变 GPU 数量、型号、依赖版本或负载后，应重新记录耗时，并将其视作当前运行环境下的测量。正式完整训练仍需 CUDA；CPU 可用于上述合成单元测试。

## 为自己的新运行采集环境

下面的命令在执行后生成当前运行的环境记录。压缩包不预置这些输出；记录只描述当前运行环境，不代表论文原实验环境。

```bash
python - <<'PY'
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import torch

destination = Path('reproduction/validation')
destination.mkdir(parents=True, exist_ok=True)
packages = ['torch', 'torch-geometric', 'numpy', 'pandas', 'scipy',
            'scikit-learn', 'matplotlib', 'networkx', 'pytest']
versions = {}
for name in packages:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
record = {
    'recorded_at': datetime.now(timezone.utc).isoformat(),
    'scope': 'This new reproduction run; not a historical environment claim',
    'python': sys.version,
    'platform': platform.platform(),
    'machine': platform.machine(),
    'versions': versions,
    'torch_build': torch.__version__,
    'cuda_runtime': torch.version.cuda,
    'cuda_available': torch.cuda.is_available(),
    'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'gpus': [dict(index=i, name=torch.cuda.get_device_name(i),
                 total_memory_bytes=torch.cuda.get_device_properties(i).total_memory)
             for i in range(torch.cuda.device_count())],
}
try:
    query = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
         '--format=csv,noheader'], capture_output=True, text=True, check=False)
    record['nvidia_smi'] = dict(returncode=query.returncode,
                               stdout=query.stdout, stderr=query.stderr)
except FileNotFoundError:
    record['nvidia_smi'] = 'not installed'
(destination / 'new_environment.json').write_text(
    json.dumps(record, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
frozen = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
(destination / 'new_pip_freeze.txt').write_text(frozen, encoding='utf-8')
print('Saved the new run environment under', destination)
PY
```

更换环境后请使用新的解压目录运行；不要把不同环境下产生的部分训练结果混合续跑。
