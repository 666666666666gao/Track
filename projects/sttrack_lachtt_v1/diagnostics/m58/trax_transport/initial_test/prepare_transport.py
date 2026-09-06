"""Create procedural RGB/depth transport fixtures; no dataset images or weights."""
from pathlib import Path
import hashlib
import json
import sys

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
INTERFACE = Path('/root/autodl-tmp/sttrack_m58_evaluation_preparation_20260906')
PARENT = Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v2_20260906')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    sys.path.insert(0, str(INTERFACE))
    from initialization_text import initialization_key, vot_wire_bbox
    evidence = json.loads((INTERFACE / 'cpu_contract_result.json').read_text())
    for name, digest in evidence['source_sha256'].items():
        assert sha(INTERFACE / name) == digest
    for channel in ['color', 'depth']:
        (ROOT / channel).mkdir()
    yy, xx = np.mgrid[:64, :96]
    images = []
    for index in range(3):
        rgb = np.stack([(xx + 20 * index) % 256, (2 * yy + 30 * index) % 256,
                        (3 * (xx + yy) + 40 * index) % 256], axis=-1).astype(np.uint8)
        depth = (500 + xx * (5 + 2 * index) + yy * (11 - 3 * index)
                 + (xx * yy) % (13 + 7 * index)).astype(np.uint16)
        color_path = ROOT / 'color' / ('%08d.jpg' % (index + 1))
        depth_path = ROOT / 'depth' / ('%08d.png' % (index + 1))
        assert cv2.imwrite(str(color_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        assert cv2.imwrite(str(depth_path), depth)
        images.append(dict(color=str(color_path), depth=str(depth_path),
                           color_sha256=sha(color_path), depth_sha256=sha(depth_path)))
    protocol = dict(scope='Procedural CPU transport fixture, not model-generated target captions.',
                    tokens='Deterministic synthetic vectors', model_weights_loaded=False)
    (ROOT / 'synthetic_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    boxes = [[1.1, 2.2, 30.3, 40.4], [19.2, 7.1, 20.7, 15.4], [40.1, 9.2, 12.3, 20.4]]
    wire = [vot_wire_bbox(box) for box in boxes]
    tokens = torch.zeros(2, 5, 768)
    for i in range(2):
        tokens[i, :3] = torch.arange(3 * 768).reshape(3, 768).float() / 1024 + i
    mask = torch.tensor([[True, True, True, False, False]] * 2)
    bank = dict(format='initialization_observation_v1', protocol_sha256=sha(ROOT / 'synthetic_protocol.json'),
        keys=[initialization_key(images[0]['color_sha256'], b) for b in wire[:2]],
        tokens=tokens, mask=mask, empty=torch.zeros(768))
    torch.save(bank, ROOT / 'synthetic_bank.pt')
    cases = []
    for index, name in enumerate(['first_object', 'second_object_same_image', 'unregistered_object']):
        predictions = []
        for step, score in enumerate([.533012345, .060412345], start=1):
            box = [wire[index][0] + .123456789 * step, wire[index][1] + .345678912 * step,
                   wire[index][2] + .111111119 * step, wire[index][3] + .222222229 * step]
            predictions.append(dict(bbox=box, wire_bbox=vot_wire_bbox(box), score=score))
        cases.append(dict(name=name, raw_bbox=boxes[index], wire_bbox=wire[index],
            expected_key_index=index if index < 2 else None, predictions=predictions))
    integration = json.loads((PARENT / 'integration.json').read_text())
    fixture = dict(scope='Synthetic transport only', images=images, cases=cases,
        bank_path=str(ROOT / 'synthetic_bank.pt'), bank_sha256=sha(ROOT / 'synthetic_bank.pt'),
        protocol_sha256=sha(ROOT / 'synthetic_protocol.json'),
        interface_sha256=evidence['source_sha256'], parent_integration_sha256=sha(PARENT / 'integration.json'),
        depth_utility_sha256=integration['source_sha256']['lib/train/dataset/depth_utils.py'],
        source_sha256={n:sha(ROOT / n) for n in ['prepare_transport.py', 'transport_worker.py', 'check_transport.py']},
        opencv_version=cv2.__version__, numpy_version=np.__version__, torch_version=torch.__version__)
    (ROOT / 'fixture.json').write_text(json.dumps(fixture, indent=2) + '\n')
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='procedural_transport_inputs_ready', fixture_sha256=sha(ROOT / 'fixture.json'),
        cases=len(cases), rgb_images=3, depth_images=3, bank_bytes=(ROOT / 'synthetic_bank.pt').stat().st_size,
        dataset_images_used=0, model_weights_loaded=False, cuda_initialized=False)))


if __name__ == '__main__':
    main()
