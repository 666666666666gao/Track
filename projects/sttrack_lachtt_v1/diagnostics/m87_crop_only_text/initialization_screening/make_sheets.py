from pathlib import Path
import json,hashlib
from PIL import Image,ImageDraw,ImageFont

R=Path(__file__).parent;W=R.parent
OLD=W.parent/'sttrack_m86_initialization_audit_20260921'
old=json.loads((OLD/'reviewed_register.json').read_text(encoding='utf-8'))
new=[json.loads(x) for x in (W/'prepared_evidence/captions/records.jsonl').read_text().splitlines()]
assert len(new)==len(old)==152
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',21)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
sheets=R/'sheets';sheets.mkdir(exist_ok=True)
rows=[]
for start in range(0,152,8):
    canvas=Image.new('RGB',(1200,1320),'#edf0f5');draw=ImageDraw.Draw(canvas)
    for offset,(a,b) in enumerate(zip(old[start:start+8],new[start:start+8])):
        assert a['sequence']==b['sequence'] and a['image_sha256']==b['image_sha256']
        x=(offset%2)*600;y=(offset//2)*330
        draw.rectangle((x+4,y+4,x+596,y+326),fill='white')
        draw.text((x+15,y+12),a['audit_id']+' | '+b['category'],fill='black',font=font)
        for path,box in [(a['anonymous_full_image'],(x+10,y+50,365,260)),(a['anonymous_target_crop'],(x+390,y+65,195,235))]:
            p=OLD/'evidence'/path;im=Image.open(p).convert('RGB')
            if path==a['anonymous_full_image']:
                bx,by,bw,bh=a['protocol_bbox'];ImageDraw.Draw(im).rectangle((bx,by,bx+bw,by+bh),outline='red',width=3)
            scale=min(box[2]/im.width,box[3]/im.height)
            im=im.resize((round(im.width*scale),round(im.height*scale)))
            canvas.paste(im,(box[0]+(box[2]-im.width)//2,box[1]+(box[3]-im.height)//2))
        rows.append(dict(audit_id=a['audit_id'],sequence=b['sequence'],category=b['category'],image_sha256=b['image_sha256'],
            full_image=str(OLD/'evidence'/a['anonymous_full_image']),crop=str(OLD/'evidence'/a['anonymous_target_crop'])))
    canvas.save(sheets/('%02d.jpg'%(start//8+1)),quality=94)
(R/'private_mapping.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(dict(rows=len(rows),sheets=19,caption_records_sha256=hashlib.sha256((W/'prepared_evidence/captions/records.jsonl').read_bytes()).hexdigest())))
