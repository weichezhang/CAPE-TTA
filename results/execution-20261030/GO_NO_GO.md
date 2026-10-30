# CAPE-TTA 当前实验判断（2026-10-30）

**NO-GO：暂不扩大 suite；原因是证据不足，而非证明 CAPE 无效。**

本轮实际完成：重新校验6个真实历史实验包，46项 manifest 校验通过；
原始结果按字节归档至 `results/verified/`。新增 CPU 校准与缓存验证逻辑，
总计53项软件单元测试通过。测试中的合成夹具只验证软件，不是实验结果。

本轮实际启动了一次新 GPU runner，但在模型加载前因缺少 torch 停止。
当前环境也没有 nvidia-smi；打开的 Colab 页面显示登录入口，没有已认证运行时。
`gpu.log`、`gpu_status.json`、`availability.json` 保存了失败和资源检查。
**本轮新完成的 VLA/GPU 实验数为0。**

已有真实证据：task0 三个开发状态存在正的 action-consequence Oracle
短程 margin；完整轨迹干预既有改善也有变慢。task1/init0 的独立 baseline
回放均在116步成功，候选评估未完成。这不是参数 adaptation 的收益信号。

尚未完成：V9六个episode的真实动作缓存和CPU replay；真实 LoRA 更新、
adapted分支；独立 benefit train/calibration/test；coverage、acceptance、
mean benefit、HIP、HAR 对比。所有这些指标当前应标记“未测”，不能填0。

最低算力下一步：运行 V9 固定 2task×3init、K4、H12、单seed。
先完整保存动作bank，再尝试rank2单步LoRA；OOM保留缓存并停止更新。
CPU独立fresh replay全部候选，核验frozen与adapted各自闭环生成轨迹。
若仅动作分支可用，只报告action-consequence pilot。

六个episode按2/2/2拆分后，alpha=.1的one-sided split conformal必然
q=+infinity、全部拒绝。没有accept时HAR未定义，不能写HAR=0或风险已受控。
不要将候选或frame充作独立calibration样本。至少需要9个独立calibration
episode才可能有有限q；是否值得扩样取决于先观察到的真实参数收益与可学习性。

可以进入论文：准确标识为开发/可行性结果的真实采样诊断、fresh-replay
确定性、已有action-consequence结果及完整轨迹局限。可以描述未运行的V9
协议为未来/计划实验，不能写为已完成实验。

绝对不能写入正式结果：smoke/preflight当benchmark表现；单元测试数据；
V9尚未运行的输出；Oracle动作替换当参数TTA；短程soft potential当官方
成功率；三个开发状态当held-out总体效果；未校准LCB；HAR<=alpha保证；
没有接收adaptation时的HAR=0；已达CCF A或保证接收。
