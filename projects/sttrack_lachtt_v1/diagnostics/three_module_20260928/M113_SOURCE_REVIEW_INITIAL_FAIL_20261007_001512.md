M113 原始源码审查：FAIL（失败证据保留）

原始 train_m113_human_initialization.py:17 使用 r['strata']==group，而当前实际 schema 是标签列表。对已有 M110 generic/Empty 的 495 条开发行执行原始 paired_vs_empty，实际得到 AssertionError('healthy')；其中 healthy 成员 264 条、transition 成员 127 条。

该错误发生在完整训练后的配对报告阶段，位于 final.pt/result.json 写入之前。执行方随后只将条件改为 group in r['strata']。此文件保留原始失败，不表示修正后仍失败。

原始文件完整 SHA 未在修改前采集，因此这里不伪造旧版本哈希。当前源码接受结论与哈希见本轮 M113_SOURCE_REVIEW.json。审查为 gpt-6-astra/max、same-family、provisional；未调用 SSH、GPU 或修改训练源码。
