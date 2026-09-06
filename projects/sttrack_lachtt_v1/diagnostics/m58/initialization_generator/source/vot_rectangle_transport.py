"""Offline rectangle input conversion for the observed TraX 4.0.2 protocol."""
import struct


def rectangle_wire_text(bbox):
    stored = struct.unpack('>4f', struct.pack('>4f', *bbox))
    return ','.join(format(value, '.4f') for value in stored)


def rectangle_wire_bbox(bbox):
    """Received xywh after float32 storage, four-decimal text, float32 parsing.

    Use only to prepare VOT initialization inputs from the toolkit's xywh.
    OPE coordinates and the tracker's internal/output boxes are not changed.
    """
    values = [float(part) for part in rectangle_wire_text(bbox).split(',')]
    return list(struct.unpack('>4f', struct.pack('>4f', *values)))
