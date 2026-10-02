"""Package existing CDTB/VOT initialization evidence for model and human review.

No caption generation, annotation import, training, or remote mutation is performed.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import subprocess
import textwrap
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader
import pymupdf

BATCH_SIZE = 32
CSV_FIELDS = [
    'dataset', 'anchor_number', 'case_id', 'sequence', 'init_frame',
    'init_frame_index', 'sequence_frames', 'batch_id', 'pdf_page',
    'provisional_category', 'provisional_attributes', 'image_path',
    'overview_path', 'video_path', 'video_start_seconds', 'source_board_sha256',
    'model_reviewer', 'model_image_reviewed', 'model_status', 'model_category',
    'model_init_stable_attributes', 'model_uncertain_attributes',
    'model_context_only_notes', 'model_evidence', 'human_reviewer',
    'human_status', 'human_category', 'human_init_stable_attributes',
    'human_uncertain_attributes', 'human_notes', 'human_updated_at',
]

PROMPT = '''你是一名 RGB-D 跟踪初始化描述审核员。请读取压缩包里的 README 和总表，再按批次实际查看图片，核对红框指定的实例。DepthTrack 已由用户审核，本包只含 CDTB 和 VOT。

一、材料与身份
1. case_id 是唯一身份。VOT 同一序列有多个不同的合法初始化点，必须逐个处理；不能只看第一条后复制整条视频的答案。
2. 每张 JPG 上方是合法初始化帧、红框和目标放大图，下方是邻近四帧。overview 是整段视频的无框代表帧。原自动类别和属性只是待核对的建议，不能直接当真值。
3. 图页底部邻近帧和全序列代表帧只能辅助辨认红框里的对象；不能追踪附近更显眼的其他物体。若无法确认多帧中的对应关系，标 uncertain，不凭序列名猜测。
4. 视频供人进一步复核。若当前环境不能看视频，明确说明；用独立 JPG 图页和代表帧审核。只有 PDF 文字提取不算看图。

二、核对内容
1. 类别：用简短英文类别，允许 object / unknown，避免精细类别猜测。
2. 稳定属性：只写初始化帧里得到图像支持、具有身份区分力的颜色、纹理、标记、形状。把手在左/右、屏幕当前内容等可能是视角或临时状态，不能无条件视为永久身份属性。
3. 支持、冲突、不可确定分开。小目标、模糊、遮挡时留空或 unknown，不把不可见视为冲突。
4. 如果只能靠后续帧才识别出对象或属性，把依据写在 model_context_only_notes；不要把后续才出现的属性放进 model_init_stable_attributes。类别若依赖多帧辨认，也要明确注明。

三、必须实际看图
请先说明：解包是否成功、本批图片是否真正能看、当前读取了多少案例。
若能运行代码，可解包后把每张 JPG 作为图像显示并查看；单纯列文件名、OCR、读取 caption 或 PDF 文本不算视觉审核。
看过图片填 model_image_reviewed=yes；无法实际看图填 no、model_status=not_visualized，并把类别、稳定属性留空。不能假装已经审核。
网页上传或上下文不能容纳全部1845条时，依照 batch_index.csv 顺序处理小批，保留上批输出。不因用户一次上传了总包就宣称全部审核完成。单批最多32条，还可按8条逐次查看图片。

四、输出
复制本批 review_template.csv 的所有行和全部列，保持 dataset、anchor_number、case_id、sequence、init_frame、路径及原自动描述不变，只填写 model_* 列。
model_reviewer 填实际模型名称/版本或你的自述；model_status 取 supported / corrected / uncertain / not_visualized。
model_category：图像支持的简短英文类别；model_init_stable_attributes 与 model_uncertain_attributes 用分号分隔。
model_evidence：简述红框内的可见依据、是否需要多帧、不确定原因及初始描述有什么错误。不得给未经校准的“准确率”。
所有 human_* 字段保持空白，human_status 保持 pending。模型初审不是独立人工真值。
输出一个 UTF-8 CSV 下载文件（能生成文件时），命名为本批编号_reviewed.csv。不能生成文件时输出完整 CSV 代码块，不省略任何行。
末尾报告输入行数、输出行数、实际看图行数、supported/corrected/uncertain/not_visualized 数量，以及尚未处理的批次。
不要按类别合并不同初始化点，不要删除重复的类别，不用想象补齐遗漏案例。

五、边界
这些是外部评测数据。审核只服务于描述可靠性检查和人工复核，不能用后续帧或外部标签训练 DepthTrack 模型、选择参数或伪装成测试期合法初始化输入。
'''

README = '''# CDTB / VOT：GPT 初审与人工复核包

本包一次性收齐 CDTB 80 条序列和 VOT-RGBD2022 1765 个合法初始化点（127 条序列），共1845条。DepthTrack 已由用户审核，本包不重复包含。原自动描述均保持原样，所有模型/人工审核结果尚未填写。

## 先交给网页版 GPT

1. 先解压总 ZIP。复制 `GPT审核提示词.txt`，提供 `batch_index.csv` 和需要审核批次的 `review_template.csv`。
2. 每个批次位于 `batches/CDTB_001` 或 `batches/VOT_001` 等文件夹。一次最多32条；可以让 GPT 按8条连续看图，并持续输出 CSV。总共有59个批次，最后CDTB一批16条、VOT一批5条。
3. 优先上传 `images/*.jpg` 这些独立图页，以及对应 `human_media/overviews/数据集/序列.jpg`。PDF 便于整本阅读，但不要假定账户能读取 PDF 内的图片。
4. 如果网页版允许解包 ZIP 和显示里面的图像，可以先上传这个总包，让 GPT 逐批处理。必须要求它真正显示、查看 JPG，并报告完成数量。若它只会读文本，请解压后直接上传独立 JPG。ZIP 是否能处理取决于当前网页工具，不能保证一次上传就完整看完1845条。
5. 把每批模型填写的 CSV 保存下来。不能实际看图时必须标记 `not_visualized`，不能拿自动描述冒充审核结果。

## 然后由你人工复核

1. 双击 `人工审核.html`（必须先解压，保留目录结构）。无需服务器或网络。
2. 填写审核人代号，选择 CDTB 或 VOT，导入 GPT 返回的 CSV。导入只更新 `model_*` 栏，不会自动填写人工确认。
3. 逐条核对初始化红框、目标放大图、前后帧、整段代表帧与视频。VOT 会显示本次初始化时刻，可以按按钮跳回该位置。
4. 可以复制模型建议，再按自己的判断修改并选“已确认 / 人工修正 / 仍无法确定”，保存。
5. 及时点“导出全部审核 CSV”。浏览器本地保存只属于当前浏览器/文件位置；多人审核请分别导出、带上各自审核人代号，不会自动在线合并。

## 看图方法与边界

- 红框只在合法初始化帧出现；后续帧和代表帧没有后续 GT 框，可能出现同类干扰物。
- 四张邻近帧分别取初始化索引 -20、-5、+5、+20，超出首尾则截到边界。
- 预览视频重采样到5 fps、宽320像素。25 fps 是序列索引换算的审核播放设置，不是已知的原始采集帧率；它不是原始全分辨率、逐帧无损视频。
- 全序列代表帧来自这些预览视频；小标记和细节以初始化放大图为准，不要放大后臆测。
- 初始化可支持属性和仅后续可见线索分别填写；后续帧不属于模型测试期合法初始化输入。
- CDTB / VOT 为外部评测数据。审核包只用于审核/诊断，不能并入 DepthTrack 训练标签，也不能靠这份审核材料选择测试集专属参数。

## 表格字段

`dataset + case_id` 绑定唯一初始化点；来源字段只读。`model_*` 是大模型初审建议；`human_*` 是你独立检查后的记录。类别用英文短词，属性分号分隔。空白表示未处理，不是审核通过。

`review_all_1845.csv` 是全量空白总表；每批有相同列的局部表。`source_manifest.json` 记录原资料哈希与范围；`package_manifest.json` 记录包内文件哈希、批次数和覆盖范围。

官方上传说明： https://help.openai.com/en/articles/8555545-file-uploads-faq
已核对的说明中：单文件上限512MB，图片20MB；PDF视觉读取存在账户方案限制，其他文件读取方式可能只提取文本。因此独立JPG是本包的主要视觉入口。实际网页工具能力变化时，以你账户提示为准。
'''


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sequence_overview(job):
    ffmpeg, source, target, sequence = job
    probe = str(Path(ffmpeg).with_name('ffprobe.exe'))
    metadata = json.loads(subprocess.run(
        [probe, '-v', 'error', '-show_entries', 'stream=nb_frames,r_frame_rate',
         '-of', 'json', str(source)], check=True, capture_output=True).stdout)
    stream = metadata['streams'][0]
    count = int(stream['nb_frames'])
    n, d = map(int, stream['r_frame_rate'].split('/'))
    fps = n / d
    positions = sorted({round(k * (count - 1) / 11) for k in range(12)})
    selector = '+'.join(f'eq(n\\,{i})' for i in positions)
    raw = subprocess.run(
        [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-threads', '1',
         '-i', str(source), '-vf',
         f'select={selector},scale=320:240:force_original_aspect_ratio=decrease,pad=320:240:(ow-iw)/2:(oh-ih)/2',
         '-fps_mode', 'passthrough', '-frames:v', str(len(positions)),
         '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
        check=True, capture_output=True).stdout
    assert len(raw) == len(positions) * 320 * 240 * 3
    board = Image.new('RGB', (1280, 904), '#152233')
    draw = ImageDraw.Draw(board)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 20)
    draw.text((20, 16), f'SEQUENCE OVERVIEW - {sequence}', fill='white', font=font)
    draw.text((20, 45), 'UNMARKED CONTEXT ONLY - sampled from a low-resolution review video', fill='#b5c5d6', font=font)
    size = 320 * 240 * 3
    for k, frame in enumerate(positions):
        tile = Image.frombytes('RGB', (320, 240), raw[k * size:(k + 1) * size])
        x, y = (k % 4) * 320, 90 + (k // 4) * 270
        board.paste(tile, (x, y))
        draw.text((x + 8, y + 242), f'Review time {frame / fps:.1f}s', fill='white', font=font)
    board.save(target, quality=85, optimize=True)
    return sequence


def image_page(source, target, dataset, row):
    board = Image.open(source).convert('RGB')
    assert board.size == (960, 565)
    out = Image.new('RGB', (960, 780), '#ffffff')
    draw = ImageDraw.Draw(out)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18)
    title = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 24)
    draw.text((20, 15), f'{dataset.upper()} #{row["number"]} | {row["sequence"]} | init {row["frame"]}', font=title, fill='#183047')
    draw.text((20, 49), 'case_id: ' + row['id'], font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15), fill='#44586c')
    caption = f'UNVERIFIED AUTO CAPTION: {row["category"]} | ' + '; '.join(row['attributes'])
    for i, line in enumerate(textwrap.wrap(caption, 94)):
        draw.text((20, 75 + i * 22), line, font=font, fill='#183047')
    assert len(textwrap.wrap(caption, 94)) <= 3
    out.paste(board, (0, 150))
    draw.text((20, 726), 'Only the initialization frame is boxed. Later frames are unmarked audit context.', font=font, fill='#44586c')
    draw.text((20, 751), 'Record uncertainty. Model review is a proposal; human confirmation is separate.', font=font, fill='#44586c')
    out.save(target, quality=88, optimize=True)


def batch_pdf(path, source_site, dataset, batch_id, batch):
    pdf = canvas.Canvas(str(path), pagesize=(841.89, 595.28), pageCompression=1)
    pdf.setTitle(batch_id + ' - initialization evidence')
    pdf.setAuthor('RGB-D initialization review packet')
    for page, r in enumerate(batch, 1):
        pdf.setFillColorRGB(.09, .19, .28)
        pdf.setFont('Helvetica-Bold', 15)
        pdf.drawString(25, 567, f'{dataset.upper()} #{r["number"]} | {r["sequence"]} | init {r["frame"]}')
        pdf.setFont('Helvetica', 8)
        pdf.drawString(25, 549, 'case_id: ' + r['id'])
        pdf.setFont('Helvetica', 10)
        caption = 'UNVERIFIED AUTO CAPTION: ' + r['category'] + ' | ' + '; '.join(r['attributes'])
        lines = textwrap.wrap(caption, 126)
        assert len(lines) <= 2
        for i, line in enumerate(lines):
            pdf.drawString(25, 534 - i * 13, line)
        pdf.drawImage(ImageReader(str(source_site / r['board'])), 25, 52, width=791.89, height=466.37)
        pdf.setFont('Helvetica', 8)
        pdf.drawString(25, 36, 'Red box = legal initialization target. Other frames are unmarked review context, not test-time input.')
        pdf.drawString(25, 22, 'Separate initialization-supported attributes from observations only visible later.')
        pdf.drawRightString(815, 22, f'{batch_id} | page {page}/{len(batch)}')
        pdf.showPage()
    pdf.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ffmpeg', required=True)
    parser.add_argument('--qa', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Use a new output path; existing review results must not be overwritten.'
    args.output.mkdir(parents=True)
    args.qa.mkdir(parents=True, exist_ok=True)
    sources, jobs, batches, all_rows, source_hashes = {}, [], [], [], []
    for dataset, expected, seq_expected in [('cdtb', 80, 80), ('vot', 1765, 127)]:
        path = args.site / 'data' / (dataset + '.json')
        source = json.loads(path.read_text(encoding='utf-8'))
        assert source['count'] == expected and source['sequence_count'] == seq_expected
        source_rows = source['rows']
        assert len(source_rows) == expected and len({r['id'] for r in source_rows}) == expected
        assert [r['number'] for r in source_rows] == list(range(1, expected + 1))
        sources[dataset] = source
        data_dir = args.output / 'sources'
        data_dir.mkdir(exist_ok=True)
        shutil.copyfile(path, data_dir / path.name)
        source_hashes.append({'dataset': dataset, 'json_sha256': sha256(path),
                              'caption_plan_sha256': source['caption_plan_sha256'],
                              'caption_records_sha256': source['caption_records_sha256'],
                              'cases': expected, 'sequences': seq_expected})
        for media in ['videos', 'overviews']:
            (args.output / 'human_media' / media / dataset).mkdir(parents=True)
        seq_rows = {r['sequence']: r for r in source_rows}
        for sequence, r in seq_rows.items():
            video = args.site / r['video']
            target = args.output / 'human_media' / 'videos' / dataset / (sequence + '.mp4')
            shutil.copyfile(video, target)
            overview = args.output / 'human_media' / 'overviews' / dataset / (sequence + '.jpg')
            jobs.append((args.ffmpeg, video, overview, sequence))
        for start in range(0, expected, BATCH_SIZE):
            number = start // BATCH_SIZE + 1
            batch_id = dataset.upper() + f'_{number:03d}'
            batch = source_rows[start:start + BATCH_SIZE]
            folder = args.output / 'batches' / batch_id
            (folder / 'images').mkdir(parents=True)
            rows = []
            for page, r in enumerate(batch, 1):
                image_path = f'batches/{batch_id}/images/{dataset}_{r["number"]:04d}_{r["id"][:12]}.jpg'
                record = dict.fromkeys(CSV_FIELDS, '')
                record.update(dataset=dataset, anchor_number=r['number'], case_id=r['id'],
                              sequence=r['sequence'], init_frame=r['frame'], init_frame_index=r['frame_index'],
                              sequence_frames=r['sequence_frames'], batch_id=batch_id, pdf_page=page,
                              provisional_category=r['category'], provisional_attributes='; '.join(r['attributes']),
                              image_path=image_path, overview_path=f'human_media/overviews/{dataset}/{r["sequence"]}.jpg',
                              video_path=f'human_media/videos/{dataset}/{r["sequence"]}.mp4',
                              video_start_seconds=r['video_start_seconds'], source_board_sha256=sha256(args.site / r['board']),
                              model_status='pending', human_status='pending')
                image_page(args.site / r['board'], args.output / image_path, dataset, r)
                rows.append(record)
            write_csv(folder / 'review_template.csv', CSV_FIELDS, rows)
            batch_pdf(folder / 'review.pdf', args.site, dataset, batch_id, batch)
            all_rows.extend(rows)
            batches.append({'batch_id': batch_id, 'dataset': dataset, 'first_number': batch[0]['number'],
                            'last_number': batch[-1]['number'], 'cases': len(batch),
                            'folder': f'batches/{batch_id}', 'pdf': f'batches/{batch_id}/review.pdf',
                            'csv': f'batches/{batch_id}/review_template.csv'})
            print('BATCH_READY', batch_id, len(batch), flush=True)
    assert len(batches) == 59 and len(all_rows) == 1845
    assert len({r['case_id'] for r in all_rows}) == 1845
    print('SEQUENCE_OVERVIEWS_START', len(jobs), flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(sequence_overview, jobs), 1):
            if i % 40 == 0:
                print('SEQUENCE_OVERVIEWS_READY', i, '/', len(jobs), flush=True)
    write_csv(args.output / 'review_all_1845.csv', CSV_FIELDS, all_rows)
    write_csv(args.output / 'batch_index.csv', list(batches[0]), batches)
    (args.output / 'GPT审核提示词.txt').write_text(PROMPT, encoding='utf-8')
    (args.output / 'README_先读.md').write_text(README, encoding='utf-8')
    template = Path(__file__).with_name('external_review_human.html').read_text(encoding='utf-8')
    embedded = json.dumps({'csv_fields': CSV_FIELDS, 'rows': all_rows}, ensure_ascii=False).replace('<', '\\u003c')
    (args.output / '人工审核.html').write_text(template.replace('__REVIEW_DATA__', embedded), encoding='utf-8')
    write_json(args.output / 'source_manifest.json', {
        'created_at': datetime.now().astimezone().isoformat(), 'datasets': source_hashes,
        'depthtrack_included': False, 'initialization_cases': 1845, 'batch_size': 32,
        'model_review_status': 'not_started', 'human_review_status': 'not_started',
        'external_dataset_training_labels': False,
        'nearby_frame_offsets': [-20, -5, 5, 20],
        'video_preview_fps': 5, 'review_index_to_time_fps': 25,
        'review_index_to_time_fps_is_not_measured_capture_rate': True})
    pages = 0
    for batch in batches:
        pdf = PdfReader(args.output / batch['pdf'])
        rows = all_rows[pages:pages + batch['cases']]
        assert len(pdf.pages) == batch['cases']
        for page, row in zip(pdf.pages, rows):
            assert row['case_id'] in page.extract_text()
        pages += len(pdf.pages)
    assert pages == 1845
    for row in all_rows:
        for field in ['image_path', 'overview_path', 'video_path']:
            assert (args.output / row[field]).is_file()
        assert row['model_status'] == row['human_status'] == 'pending'
    samples = [('CDTB_001', 0), ('CDTB_003', 15), ('VOT_001', 0), ('VOT_028', 15), ('VOT_056', 4)]
    rendered = []
    for batch_id, page_number in samples:
        pdf = pymupdf.open(args.output / 'batches' / batch_id / 'review.pdf')
        image = args.qa / f'{batch_id}_page_{page_number + 1:02d}.png'
        pdf[page_number].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(image)
        rendered.append(str(image))
        pdf.close()
    manifest = {'cases': 1845, 'datasets': {'cdtb': 80, 'vot': 1765},
                'sequences': {'cdtb': 80, 'vot': 127}, 'batches': 59,
                'pdf_pages': pages, 'images': 1845, 'videos': 207, 'overviews': 207,
                'files': []}
    for path in sorted(args.output.rglob('*')):
        if path.is_file():
            manifest['files'].append({'path': path.relative_to(args.output).as_posix(),
                                      'bytes': path.stat().st_size, 'sha256': sha256(path)})
    write_json(args.output / 'package_manifest.json', manifest)
    write_json(args.qa / 'generation_check.json', {k: v for k, v in manifest.items() if k != 'files'} | {'rendered_samples': rendered})
    archive = args.output.with_suffix('.zip')
    assert not archive.exists()
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=2) as bundle:
        for path in sorted(args.output.rglob('*')):
            if path.is_file():
                suffix = path.suffix.lower()
                compression = zipfile.ZIP_STORED if suffix in ['.jpg', '.mp4', '.pdf'] else zipfile.ZIP_DEFLATED
                bundle.write(path, (Path(args.output.name) / path.relative_to(args.output)).as_posix(), compress_type=compression)
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert len(bundle.namelist()) == len(manifest['files']) + 1
    assert archive.stat().st_size < 512 * 1024 * 1024
    receipt = {'archive': str(archive), 'bytes': archive.stat().st_size,
               'sha256': sha256(archive), 'file_count': len(manifest['files']) + 1,
               'coverage': {k: v for k, v in manifest.items() if k != 'files'},
               'visual_inspection': 'pending', 'source': str(args.site)}
    write_json(args.qa / 'delivery_receipt.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
