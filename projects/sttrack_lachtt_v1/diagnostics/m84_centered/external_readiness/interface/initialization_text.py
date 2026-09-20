"""Bind initialization text to the observed RGB bytes and the actual region."""
import hashlib
from pathlib import Path
import struct


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def initialization_key(rgb_sha256, bbox):
    # Preserve the actual four coordinates. VOT input preparation must first
    # apply its float32 wire conversion; OPE uses its original initialization.
    return hashlib.sha256(bytes.fromhex(rgb_sha256) + struct.pack('>4d', *bbox)).hexdigest()


def vot_wire_bbox(bbox):
    """The observed TraX 4.0.2 rectangle conversion, for offline VOT inputs."""
    return list(struct.unpack('>4f', struct.pack('>4f', *bbox)))


class InitializationTextBank:
    def __init__(self, path, expected_sha256, protocol_sha256):
        import torch
        assert sha(path) == expected_sha256
        bank = torch.load(path, map_location='cpu')
        assert bank['format'] == 'initialization_observation_v1'
        assert bank['protocol_sha256'] == protocol_sha256
        keys = bank['keys']
        assert len(keys) == len(set(keys))
        assert bank['tokens'].shape == (len(keys), 5, 768)
        assert bank['mask'].shape == (len(keys), 5)
        assert bank['mask'].dtype == torch.bool and bank['mask'].any(dim=1).all()
        assert bank['empty'].shape == (768,)
        assert torch.isfinite(bank['tokens']).all() and torch.isfinite(bank['empty']).all()
        self.bank = bank
        self.indices = {key: index for index, key in enumerate(keys)}

    def info(self, rgb_path, bbox):
        index = self.indices[initialization_key(sha(rgb_path), bbox)]
        return dict(init_bbox=list(bbox), text_tokens=self.bank['tokens'][index],
                    text_mask=self.bank['mask'][index], empty_text=self.bank['empty'])
