# V7 真实参数更新开发验证及后续实验协议

## 当前能执行什么
V7 是开发实验，默认只跑 task 0 / init 0。三个分支均为最多 220 policy steps：冻结模型、一次 LoRA 更新后的模型、恢复原始 adapter 后的回放。更新后第一步由更新模型生成，后续也使用更新模型，不能把 V5 的动作直接注入并称为适应。

训练目标是 V5 已保存候选的七个 token，仅监督动作 token。8-bit OpenVLA，FP16 autocast，FP32 LoRA，q_proj/v_proj，rank 4，alpha 8，dropout 0，AdamW lr 1e-5，weight decay 0，gradient clip 1。默认一个 optimizer step，不因结果调参。梯度非有限、adapter 未变化、冻结权重哈希变化、恢复后参数/动作/完整轨迹不一致均报错。失败仍打包日志。T4 上的训练显存及兼容性尚未验证。

运行成功也只证明 privileged pseudo-target 适应机制可执行。当前没有 observation-only verifier、更新收益预测器或在线门控；V7 不声称实现完整 CAPE-TTA。

## 阶段 1 通过条件
检查非零 adapter delta、冻结参数前后哈希一致、完整回合轨迹、恢复参数完全一致及回放一致。保留负更新收益。不要在 init 0 成功后把 init 1、2 改称 held-out。下一步沿用相同规则检查 1、2。

## 阶段 2 收益标签与校准
预先固定 utility = success - 0.1 * executed_steps/220。失败的 220 步是预算耗尽，不当作完成时间。标签为实际更新模型 utility 减未更新模型 utility；动作 soft margin 不能替代该标签。

冻结更新算子、候选规则和 observation-only 预测器之后，用独立校准记录计算单侧 residual = prediction - actual_update_margin；有限样本秩 ceil((n+1)(1-alpha))，秩超出样本数时拒绝全部更新。selective_update.py 实现统计核心，但没有训练好的预测器。alpha 默认 .05。必须报告 acceptance 与 harm，不能将低接受率包装成高可靠性。

在新的数据收集之前固定 split manifest。建议 task 0-3 用于开发/拟合，4-5 用于校准，6-9 用于任务外测试；每个任务先枚举合法初始状态，按任务组划分，不按候选分支划分。这个任务外划分不自动满足 conformal exchangeability，因此只报告经验覆盖，不宣称分布无关保证。若需要交换性下的理论实验，另建同分布、独立记录的校准与测试方案，并明确其适用范围。所有当前 init 0、1、2 只属于开发。

## 阶段 3 正式对照及消融
必须先完成阶段 1、2，不能现在启动未实现的方法。共享 checkpoint、精度、候选种子、候选数、环境重置与 220 步上限。比较 frozen、equal-budget random candidate、uncalibrated observation-only selector、ungated LoRA、gated LoRA 和 privileged oracle。消融分别移除校准、反事实证据、rollback；固定候选池以分离采样收益。

记录每回合 success、trajectory、实际 optimizer steps、adapter delta、update utility margin、acceptance、rollback、模型 forward/backward 次数、所有预选/标签分支计算量、wall time 和峰值显存。完整结果按 task/state/seed 配对，跨任务汇总时以任务为分组单位，不把候选分支当独立样本。正式测试前保存协议哈希；不得根据测试结果再选阈值或温度。算力预算不足时报告样本不足，不填缺失数字。

## 论文写回门槛
只有上述实际结果齐全后，才改写摘要为有效性结论；若没有收益，保留负结果并调整论文定位。代码通过 CPU 测试不等于机器人适应成功。
