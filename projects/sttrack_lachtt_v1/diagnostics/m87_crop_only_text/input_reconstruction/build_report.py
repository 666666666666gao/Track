from pathlib import Path
import hashlib,html,json,shutil
R=Path(__file__).resolve().parent;W=R.parent;E=W/'input_audit_evidence'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
result=read(E/'output/result.json');rows=read(E/'output/rows.json')
assert result['rows']==152 and all(n==152 for n in result['checks'].values())
assert result['generator_calls']==result['model_weight_loads']==result['optimizer_steps']==0
report='''# M87 图像预处理CPU重建核查

152条已冻结初始化输入的预处理重建已完成，真实退出0。未加载Qwen权重、未调用generate、未使用GPU、未训练，也未修改M87文本或推理策略。本核查不提供新的跟踪指标。

| 核查项 | 通过 / 总数 |
|---|---:|
| 原图SHA与冻结记录一致 | 152 / 152 |
| 目标裁剪尺寸与历史记录一致 | 152 / 152 |
| 重建image_grid_thw与历史生成记录一致 | 152 / 152 |
| 单图占位及实际image token数量一致 | 152 / 152 |
| 图像tensor全部有限 | 152 / 152 |
| patch逆排列、反归一化及取整后RGB逐像素一致 | 152 / 152 |
| 静态图的两个时间patch逐值相同 | 152 / 152 |

实际image token为128–180；原始紧裁剪短边的最小值、中位数、最大值为16、57、349像素。反归一化浮点最大误差为4.57763671875e-05个8位像素单位，取整后所有RGB像素完全一致。图像经过尺寸调整，因此该逐像素比较对象是预处理入口的已缩放RGB图，不是原始未缩放裁剪。

逆变换依据本机实际安装的Qwen2VLImageProcessor代码，先恢复时空patch排列，再撤销mean/std与rescale；token数量根据同一模型config中的image_token_id与spatial_merge_size验证。使用Qwen环境Python3.10、torch2.5.1+cu121、transformers4.51.3，但CUDA_VISIBLE_DEVICES为空且没有加载生成模型。环境源码与相关模型配置均保存SHA绑定。

## 可以与不能得出的结论

当前相同绑定输入的CPU重建没有发现空图、RGB通道颠倒、patch错位或图像占位数量不符。8项预定可视化包含篮球、书本/前景遮挡、暗猫、杯子、鸭、蛋、细小手持物和滑板；重建图保留相应原始可见信息，没有为文字匹配重画对象。

历史生成没有保存pixel_values，所以本次不是历史内部tensor逐字节回放。历史grid、图像SHA、裁剪尺寸、caption源码及processor配置可以交叉验证；当时模型内部视觉注意力、数值状态及生成语义未被本实验重放或审计。库源码哈希是本次记录，不能回溯证明历史安装未变。

因此不能把预处理检查通过写成生成器输入链完全无误或caption错误根因已经确定。小裁剪、缺少上下文、遮挡和模型辨识能力都是仍需隔离的因素。尤其不能把放大后的像素数当成新增目标细节，也不能根据本诊断中途改写M87训练文本。

本次重建结果与初始化语义初筛是两层证据：前者验证当前输入处理，后者判断可见类别相容性；二者均不能替代同权重内容对照和完整递归评测。

## 文件绑定

'''
for key in ['script_sha256','plan_sha256','caption_spec_sha256','caption_records_sha256','output_rows_sha256']:
    report+='- '+key+': `'+result[key]+'`\n'
report+='- 私有证据包SHA256：`fa76a71ba7fc90ff8112e4eac94b14c2fb9ed758c72076ce1843f385a225510e`\n'
(R/'M87_INPUT_RECONSTRUCTION.md').write_text(report,encoding='utf-8')
out=Path(r'C:\Users\gb\Desktop\document\RGBD_TEXT_PROTOCOL_COMPARISON_20260921/input_reconstruction')
out.mkdir();shutil.copytree(E/'output/images',out/'images')
for source,name in [(R/'M87_INPUT_RECONSTRUCTION.md','M87_INPUT_RECONSTRUCTION.md'),(E/'output/result.json','result.json'),(E/'output/rows.json','rows.json')]:shutil.copyfile(source,out/name)
cards=[]
for row in rows:
    if row['audit_id'] not in result['selected_visual_ids']:continue
    aid=row['audit_id'];raw='images/'+aid+'_crop.png';processed='images/'+aid+'_reconstructed.png'
    assert (out/raw).is_file() and (out/processed).is_file()
    cards.append('<article><h2>%s · %s · 自动类别 %s</h2><div><figure><img src="%s"><figcaption>原始紧裁剪 %s</figcaption></figure><figure><img src="%s"><figcaption>从pixel_values重建 %s</figcaption></figure></div><p>历史grid匹配；%d个image token；RGB取整后逐像素一致。</p></article>'%(aid,html.escape(row['sequence']),html.escape(row['category']),raw,str(row['crop_size']),processed,str(row['resized_size']),row['image_tokens']))
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>M87 输入预处理重建</title><style>body{font:16px/1.6 system-ui;background:#f3f5f8;color:#17212b;max-width:1200px;margin:30px auto;padding:0 20px}article{background:white;padding:20px;margin:20px 0;border-radius:10px}article div{display:flex;flex-wrap:wrap}figure{flex:1;min-width:220px;margin:10px}img{width:100%;height:280px;object-fit:contain;background:#e9edf2}h2{font-size:19px}</style><h1>M87 输入预处理重建</h1><p>152项CPU检查完成；以下8项按核查计划展示。不是历史生成重放，没有生成新caption，没有加载Qwen权重，也不提供跟踪性能。原图放大不增加细节。</p><p><a href="M87_INPUT_RECONSTRUCTION.md">完整报告与边界</a> · <a href="result.json">实测回执</a></p>'''+''.join(cards)+'</html>'
(out/'index.html').write_text(page,encoding='utf-8')
validation=dict(cards=len(cards),image_links_verified=16,browser_tested=False,gallery_path=str(out/'index.html'),
    report_sha256=hashlib.sha256((R/'M87_INPUT_RECONSTRUCTION.md').read_bytes()).hexdigest())
(R/'artifact_validation.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8')
print(json.dumps(validation))
