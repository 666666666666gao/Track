"""Strict tracker checkpoint loading with actionable compatibility errors."""

import torch


LANGUAGE_STATE_PREFIXES = (
    'language_encoder.',
    'language_encoder_proj.',
    'language_fusion.',
)


def _is_language_key(key):
    return key.startswith(LANGUAGE_STATE_PREFIXES)


def load_tracker_checkpoint(network, checkpoint_path, language_enabled=False):
    """Load a tracker checkpoint strictly after reporting structural incompatibilities."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('net'), dict):
        raise RuntimeError("Tracker checkpoint must contain a 'net' state dictionary: {}".format(
            checkpoint_path))

    checkpoint_state = checkpoint['net']
    model_state = network.state_dict()
    missing = sorted(set(model_state) - set(checkpoint_state))
    unexpected = sorted(set(checkpoint_state) - set(model_state))
    shape_mismatches = sorted(
        key for key in set(model_state) & set(checkpoint_state)
        if tuple(model_state[key].shape) != tuple(checkpoint_state[key].shape)
    )

    if missing or unexpected or shape_mismatches:
        language_mismatch = any(
            _is_language_key(key) for key in missing + unexpected + shape_mismatches
        )
        if language_enabled and language_mismatch:
            raise RuntimeError(
                'MODEL.LANGUAGE.USE=True requires a language-trained checkpoint with matching '
                'encoder and fusion parameters. The selected checkpoint appears to be visual-only '
                'or uses a different language configuration: {}'.format(checkpoint_path))
        if not language_enabled and language_mismatch:
            raise RuntimeError(
                'The selected checkpoint contains language parameters, but MODEL.LANGUAGE.USE=False: {}'.format(
                    checkpoint_path))
        raise RuntimeError(
            'Checkpoint structure does not match the tracker (missing={}, unexpected={}, shape_mismatches={}): {}'.format(
                missing[:8], unexpected[:8], shape_mismatches[:8], checkpoint_path))

    network.load_state_dict(checkpoint_state, strict=True)
    return checkpoint
