"""Read-only provenance audit and anonymous initialization-only review sheets."""
import hashlib
import json
import shutil
import tarfile
import textwrap
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

B = Path('/root/autodl-tmp')
R = B / 'sttrack_m86_initialization_audit_20260921'
V1 = B / 'sttrack_m58_semantic_spatial_v1_20260906'
V2 = B / 'sttrack_m58_semantic_spatial_v2_20260906'
C = V1 / 'initial_captions_v2'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def main():
    assert not (R / 'audit_register.json').exists()
    out = R / 'evidence'
    out.mkdir()
    (out / 'sources').mkdir()
    (out / 'images').mkdir()
    (out / 'sheets').mkdir()
    plan, result = read(C / 'plan.json'), read(C / 'result.json')
    assert sha(C / 'plan.json') == result['plan_sha256'] == '50d83fe2811acbec045fd35dee255f634ada284aa39fc43e62142d2b36a2addb'
    assert sha(C / 'records.jsonl') == result['records_sha256'] == 'ae92f492bcc1e9ca08bed3a734f934210f0c1a9bdd212f0c701c3686d1e6286e'
    assert sha(V1 / 'caption_initial_v2.py') == plan['script_sha256']
    assert sha(V1 / 'data_inventory.json') == plan['inventory_sha256']
    records = [json.loads(s) for s in (C / 'records.jsonl').read_text().splitlines()]
    inventory = {r['sequence']: r for r in read(V1 / 'data_inventory.json')['sequences_detail']}
    manifest = {r['sequence']: r for r in read(V2 / 'text_manifest.json')}
    assert len(records) == len(inventory) == len(manifest) == len(plan['rows']) == 152
    assert [r['sequence'] for r in records] == [r['sequence'] for r in plan['rows']]
    original, current, indices = {}, {}, {}
    bindings = {}
    for split, expected in [('fit', 130), ('development', 22)]:
        op = V2 / ('text_' + split + '.pt')
        cp = B / 'sttrack_m67_supervised_semantic_support_20260907' / op.name
        original[split] = torch.load(op, map_location='cpu')
        current[split] = torch.load(cp, map_location='cpu')
        o, c = original[split], current[split]
        assert c['sequences'] == o['sequences'] and len(c['sequences']) == expected
        assert c['source_bank_sha256'] == sha(op)
        assert c['caption_records_sha256'] == o['caption_records_sha256'] == result['records_sha256']
        assert c['encoder_sha256'] == o['encoder_sha256']
        assert torch.equal(c['mask'], o['mask']) and torch.equal(c['empty'], o['empty'])
        expected_tokens = o['tokens'].clone()
        for i in range(expected):
            for k in range(1, 5):
                if bool(o['mask'][i, k]): expected_tokens[i, k] = o['empty']
        assert torch.equal(c['tokens'], expected_tokens)
        indices[split] = {s: i for i, s in enumerate(c['sequences'])}
        bindings[split] = dict(original_bank=str(op), original_sha256=sha(op), current_bank=str(cp), current_sha256=sha(cp))
    register = []
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    panels = []
    for n, (row, rec) in enumerate(zip(plan['rows'], records), 1):
        seq, split = row['sequence'], row['split']
        inv, man = inventory[seq], manifest[seq]
        assert seq == rec['sequence'] and split == rec['split'] == inv['split'] == man['split']
        image = Path(row['image'])
        assert image.name == '00000001.jpg' and str(image) == inv['initial_rgb']
        assert sha(image) == row['image_sha256'] == rec['image_sha256'] == man['first_image_sha256']
        # Read only the initialization GT line, never later tracking outcomes.
        gt_path = image.parent.parent / 'groundtruth.txt'
        with gt_path.open() as f: first_line = f.readline().strip()
        gt = [float(x) for x in first_line.replace(',', ' ').split()]
        assert gt == inv['first_box']
        im = Image.open(image).convert('RGB')
        assert list(im.size) == row['image_size']
        x, y, w, h = gt
        expected_box = [max(0, int(x)), max(0, int(y)), min(im.width, int(x+w+.999999)), min(im.height, int(y+h+.999999))]
        assert expected_box == row['target_xyxy']
        raw_path = C / (seq + '.raw.txt')
        raw = raw_path.read_text()
        assert raw == rec['raw']
        serialized = raw.strip()
        if serialized.startswith('```json\n') and serialized.endswith('\n```'): serialized = serialized[8:-4]
        parsed = json.loads(serialized)
        assert parsed == dict(category=rec['category'], attributes=rec['attributes'])
        phrases = [rec['category']] + rec['attributes']
        i = indices[split][seq]
        o, c = original[split], current[split]
        assert phrases == man['phrases'] == o['initialization_phrases'][i] == c['original_initialization_phrases'][i]
        assert c['mask'][i].tolist() == [k < len(phrases) for k in range(5)]
        aid = '%03d' % n
        shutil.copyfile(image, out / 'images' / (aid + '.jpg'))
        crop = im.crop(expected_box)
        crop.save(out / 'images' / (aid + '_crop.png'))
        record = dict(audit_id=aid, sequence=seq, split=split, bank_row=i, initialization_image=str(image),
            image_sha256=sha(image), first_gt_line=first_line, protocol_bbox=gt, generator_target_xyxy=expected_box,
            raw_response=raw, raw_response_sha256=sha(raw_path), parsed=parsed,
            original_encoding_strings=phrases, current_encoding_strings=[phrases[0]] + ['']*(len(phrases)-1),
            mask=c['mask'][i].tolist(), padding_slots=list(range(len(phrases),5)),
            anonymous_full_image='images/'+aid+'.jpg', anonymous_target_crop='images/'+aid+'_crop.png',
            crop_sha256=sha(out/'images'/(aid+'_crop.png')), binding_checks_passed=True,
            semantic_verdict='pending_initialization_only_visual_review')
        register.append(record)
        full = im.copy()
        ImageDraw.Draw(full).rectangle(expected_box, outline='red', width=2)
        full.thumbnail((320,180))
        crop.thumbnail((180,180))
        # Enlarged crop is a display only; source pixels are retained separately.
        crop = im.crop(expected_box)
        scale = min(180/crop.width,180/crop.height)
        crop = crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))), Image.Resampling.NEAREST)
        panel = Image.new('RGB',(860,224),'white');draw=ImageDraw.Draw(panel)
        draw.text((8,5),'ID '+aid+' | '+rec['category'],font=font,fill='black')
        panel.paste(full,(4,35)); panel.paste(crop,(330,35))
        txt='category: '+rec['category']+'\nattributes: '+'; '.join(rec['attributes'])
        lines=[]
        for line in txt.splitlines(): lines.extend(textwrap.wrap(line,width=35))
        draw.multiline_text((520,36),'\n'.join(lines),font=font,fill='black',spacing=4)
        draw.line((0,223,859,223),fill='gray')
        panels.append(panel)
    for start in range(0,152,8):
        sheet=Image.new('RGB',(1720,896),'#dddddd')
        for j,panel in enumerate(panels[start:start+8]):sheet.paste(panel,((j%2)*860,(j//2)*224))
        sheet.save(out/'sheets'/('sheet_%02d.png'%(start//8)))
    for source in [C/'plan.json',C/'result.json',C/'records.jsonl',V1/'caption_initial_v2.py',V1/'prepare_text.py',V2/'text_preparation.json',V2/'text_manifest.json']:
        shutil.copyfile(source,out/'sources'/source.name)
    report=dict(status='provenance_pass_visual_review_pending',rows=152,fit=130,development=22,bindings=bindings,
        source_sha256=sha(__file__),caption_plan_sha256=sha(C/'plan.json'),records_sha256=sha(C/'records.jsonl'),
        first_frame_only=True,later_gt_or_tracking_metrics_loaded=False,generation_rerun=False,
        clip_reencoded=False,model_weights_rehashed=False,
        limitation='Current source images/raw parser/bank rows and vectors checked; historical Qwen pixel tensors and CLIP generation not rerun. Semantic judgments pending.')
    for name,value in [('audit_register.json',register),('provenance_result.json',report)]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    (R/'audit_register.json').write_text(json.dumps(register,indent=2)+'\n')
    shutil.copyfile(__file__,out/'prepare_audit.py')
    files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(files,indent=2)+'\n')
    archive=R/'initialization_evidence.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for p in sorted(out.rglob('*')):
            if p.is_file():t.add(p,arcname=str(p.relative_to(out)))
    print(json.dumps(dict(**report,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size),indent=2))


if __name__=='__main__': main()
