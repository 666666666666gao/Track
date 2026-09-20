# M87 Qwen模型文件来源补充核验

预处理重建之后，单独将当前本地模型的14个文件逐一核对至Qwen官方固定revision元数据。两份权重以Git LFS SHA256核验；其余文件同时核验下载manifest的SHA256及官方Git blob SHA1。14项全部一致，总计7,520,919,614字节，包含两份safetensors、config、generation_config、chat_template、processor配置、tokenizer、merges和vocab等。

官方固定版本：[Qwen/Qwen2.5-VL-3B-Instruct @66285546](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/66285546d2b821cf421d4f5eb2576359d3770cd3)。原始API元数据另存official_model_tree.json；检查工具的网页读取器无法读取API时，使用PowerShell HTTP读取同一公开JSON并封存。没有下载或替换权重。

这确认当前文件与官方固定版本对应，不证明历史生成运行内部状态，也不证明输出语义正确。没有加载模型、调用生成器或改动M87。

- metadata SHA256: `96fe2dd92d557c09ae2a1cd050a09d22093f010f61b01e24fa54d4ccee9ee671`
- result SHA256: `ce0879de512c2fc4a42ef21c5a51fa3ce8ba78b6145a8802e1db7b743554f36b`
