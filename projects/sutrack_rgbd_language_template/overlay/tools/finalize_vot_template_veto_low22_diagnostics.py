#!/usr/bin/env python3
"""Explain the frozen low22 baseline-first template-veto result."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import finalize_vot_transaction_low22_diagnostics as base


base.SCHEMA = 'sutrack_template_veto_low22_diagnostics_v1'
base.GATE_SCHEMA = 'sutrack_template_veto_low22_gate_v1'
base.CANDIDATE_TRACKER = (
    'sutrack_l384_rgbd_anchor_identity_template_veto_low22')


if __name__ == '__main__':
    base.main()
