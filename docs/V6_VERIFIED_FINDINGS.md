# V6 已核验结果

用户上传 full-check-20261028-115543-36875ca2.zip。
ZIP SHA256: 28f643d6756b6050d5ad1a23967381c20f82b471164ab1456e2672bf2dc55595。
8 个 setup/rollout 文件校验条目均匹配，两个分支轨迹步数、连续 step 编号、第一动作与 V5 固定候选均核验一致。

Spatial/task 0/init 0，冻结模型与量化配置同 V5；source SHA 8f04f8898591add10db035b6bad08acf15823874。
wrapper parity=0，V5 baseline action mismatch=0。

| 分支 | 成功 | policy steps |
|---|---|---:|
| 冻结基线 | true | 85 |
| V5 privileged 第一动作，随后冻结贪心 | true | 74 |

本开发状态中候选减少 11 个动作步骤（相对基线 12.94%），不包含 10 个 settle steps；不等同于运行时间减少。
两者都成功，没有观察到该状态的二值成功差异。该候选由 V5 开发 Oracle 选出，不能解释为部署方法、成功率提高或 held-out 泛化。
参数更新次数=0。没有技术异常导致提前失败；程序状态 full_horizon_complete。

按已固定规则，下一阶段补测 init 1、2；不依据后续完整回合结果调整候选或温度。
