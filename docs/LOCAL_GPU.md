# 本地 GPU：运行候选多样性诊断

## 能运行的环境

本流程面向 NVIDIA CUDA GPU 的 Linux 环境；Windows 用户使用具备 CUDA
访问能力的 WSL2 Linux 环境。先在实际运行终端执行 `nvidia-smi` 确认显卡可见。
本对话的执行容器没有 NVIDIA 设备，不能调用用户电脑 GPU。
之前已验证的硬件是 Colab Tesla T4 15360 MiB，8-bit OpenVLA + FP16；
本地兼容性仍需实际验证，不能承诺所有显卡均可运行。

## 已有 V3 环境

从仓库目录运行，替换 `/absolute/path/to/LIBERO` 为真实 checkout 路径：

```bash
python scripts/probe_openvla_diversity.py --libero-root /absolute/path/to/LIBERO
```

脚本只依赖传入目录，不使用 `/content` 等 Colab 固定路径。输出保存在
`results/diversity/probe-时间-标识.zip`。模型只加载一次。

## 没有环境时（已有 conda 的 Ubuntu / WSL Ubuntu）

不要在 V3 已验证环境中盲目升级 torch、transformers 或 MuJoCo。
下面仅配置软件环境，不负责安装或修改宿主机显卡驱动。

```bash
git clone https://github.com/weichezhang/CAPE-TTA.git
cd CAPE-TTA
sudo apt-get update
sudo apt-get install -y git ffmpeg libosmesa6-dev libgl1 libglx-mesa0 libglfw3 build-essential
conda create -n cape-v3 python=3.10 pip -y
conda activate cape-v3
python -m pip install pip==24.0 setuptools==69.5.1 wheel==0.43.0
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
python -m pip install -r requirements/openvla_colab.txt
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git -C third_party/LIBERO checkout --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01
python -m pip install --no-build-isolation --no-deps -e . -e third_party/LIBERO
python -m pip check
python scripts/probe_openvla_diversity.py --libero-root third_party/LIBERO
```

如果已经克隆过仓库，先检查自己的工作区改动，再获取更新；不要重复克隆到同名目录。
无需上传密钥、授权 GPU 服务或购买算力。

## 实验含义

固定 Spatial/task 0/init 0，候选 seed 为 8、9、10、11，开发网格固定为：

| 设置 | temperature | top_p | top_k |
|---|---:|---:|---|
| 原设置 | 0.7 | 0.95 | 继承 checkpoint generation_config |
| 关闭截断 A | 0.7 | 1.0 | 0 |
| 关闭截断 B | 1.0 | 1.0 | 0 |
| 关闭截断 C | 1.5 | 1.0 | 0 |

默认只诊断一个初始状态，先解释已有候选重复现象；可显式传入
`--init-state-ids 0 1 2` 做开发扩展。不是正式测试集。
每组记录 token 重复率、执行动作重复率、与基线的动作距离，以及每个生成位置的
概率支持、最大概率和熵。概率来自 generation processors/warpers 处理后的分布；
不同温度设置下的 processed logprob 不能直接比较为原始策略置信度。
后续 token 分布依赖已采样的前缀，因此跨设置只比较分布统计时要注意条件不同。
脚本不会强制输出独立候选，也不根据结果自动选“最优”温度。

完成标记 diagnostic_complete 仅表示诊断执行完毕。
本脚本没有 rollout、没有训练、不验证成功率收益，也不是最终 CAPE-TTA。
更高温度制造的差异可能降低动作质量；获得多样性后仍需完整回合对照。

## 已核验的 V3 记录

2026-10-28 上传包 `run-20261028-045620-7e26d300.zip`：6 个文件校验和一致；
wrapper parity=0；fresh replay 状态/软进展误差=0；4 候选只有 1 个独立动作；
Oracle margin=0；12 步内没有任务成功。不将短回合结果解释为完整回合性能。
原始运行包由用户提供，本文件只保留核验摘要，不将未运行诊断写成成功。
