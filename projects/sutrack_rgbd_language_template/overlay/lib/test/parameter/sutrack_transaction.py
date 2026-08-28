"""Parameter loader for protected/tentative SUTrack template transactions."""

import os
from pathlib import Path

from lib.test.parameter.sutrack import parameters as base_parameters


def parameters(yaml_name):
    params = base_parameters(yaml_name)
    params.protected_tentative_transaction = {
        'confirm_frames': 2,
        'max_shadow_frames': 2,
        'min_confidence': 0.65,
        'min_response_margin': 0.10,
        'min_identity': 0.75,
        'min_depth_consistency': 0.50,
        'min_temporal_continuity': 0.50,
        'max_confidence_deficit': 0.05,
        'max_margin_deficit': 0.05,
        'min_utility_advantage': 0.01,
        'confidence_weight': 0.30,
        'margin_weight': 0.20,
        'identity_weight': 0.25,
        'depth_weight': 0.10,
        'temporal_weight': 0.15,
    }
    trace_root = os.environ.get('SUTRACK_TRANSACTION_TRACE_ROOT', '').strip()
    if not trace_root:
        raise ValueError(
            'SUTRACK_TRANSACTION_TRACE_ROOT must identify this run trace root')
    params.transaction_trace_root = str(Path(trace_root).resolve())
    return params


__all__ = ['parameters']
