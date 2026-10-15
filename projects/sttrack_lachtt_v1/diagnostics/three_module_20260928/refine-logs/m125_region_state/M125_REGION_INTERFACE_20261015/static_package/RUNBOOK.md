# M125区域后继执行：当前仅已暂存

静态root：`/root/autodl-tmp/m125_region_reviewed_admission_inputs_20261015`。`inputs/`含17冻结文件和原区域审查MD/JSON；`operator/`含已审R2部署操作及其审查；`code/`为实际核对的完整64源码布局。当前没有区域正式arm root、GPU real64或完整训练。

1. 唯一原R0后台等待PID49616到北京时间2026-10-17 12:45只读一次。到点后先读本地`.aris/m122_full_causal_20261007/M125_R0_training_end_observation_20261017.json`及对应timer terminal；不要重复启动等待或旧观察脚本。此日期不是新GPU启动承诺。
2. 若原R0仍在训练/评价，保留原队列，按实际剩余时间安排后继唯一节点；不提前调用新臂。原R0完整6pass/同final全部DT/CDTB/VOT和退出/收集回执都须完成，再审计其新正式结果。
3. 后继使用本机现有Python和已授权凭据环境，执行下面的私有原位置脚本。密码不写入代码、文档或版本库。下面是命令说明，当前没有执行它们。

```powershell
& 'C:/Users/gb/AppData/Roaming/uv/python/cpython-3.13-windows-x86_64-none/python.exe' -X utf8 'C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/invoke_M125_region_reviewed_phase.py' --arm roi --phase real64
```

4. 读取实际real64 launch后按预计时间只观察一次对应root的原始退出/结果，检查64track、两次AB/C更新、四新组梯度、冻结/共同warm/保存重载等。真实64权重不进入正式起点。
5. real64实际通过且资源释放后，用同脚本`--arm roi --phase full`启动固定6pass/final6及全部三集suite。无自动重试，不因SSH/观察超时重启原job。正式启动后用首个完整序列前缀速度估计收尾时间，不轮询外部指标选权重。
6. 原ROI整套终态后再按同协议独立运行language/R1和visual/V；每臂从共同warm起，不继承前臂final。状态观察R2在区域完整结果后决定如何实现；记忆价值仍为有条件研究项。

本地边界测试是在当前日期实际调用入口并被not-before拒绝，没有SSH或任务尝试文件；只能证明当时的本地边界。当前静态CPU导入也不能证明新NN或性能。源码/候选/数据、预算与同final九项要求以已审SPEC、固定协议和主文为准。
