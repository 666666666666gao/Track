#!/usr/bin/env python3
"""Seal and gate the frozen low22 baseline-first template-veto result."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import finalize_vot_transaction_low22 as base


base.TRACKER = 'sutrack_l384_rgbd_anchor_identity_template_veto_low22'
base.GATE_RESULT_SCHEMA = 'sutrack_template_veto_low22_gate_v1'
base.SOURCE_SNAPSHOT_SCHEMA = 'sutrack_template_veto_low22_sources_v1'
base.TRANSACTION_SOURCE_FILES = (
    'experiments/sutrack/'
    'sutrack_l384_rgbd_anchor_identity_template_veto_low22.yaml',
    'lib/test/parameter/sutrack_transaction.py',
    'lib/test/parameter/sutrack_template_veto_transaction.py',
    'lib/test/tracker/protected_tentative_transaction.py',
    'lib/test/tracker/sutrack_transaction.py',
    'lib/test/tracker/sutrack_template_veto_transaction.py',
    'lib/test/vot/'
    'sutrack_l384_rgbd_anchor_identity_template_veto_low22.py',
    'lib/test/vot/sutrack_transaction_class.py',
    'tools/prepare_vot_transaction_low22.py',
    'tools/prepare_vot_template_veto_low22.py',
    'tools/launch_vot_template_veto_low22.sh',
    'tools/finalize_vot_transaction_low22.py',
    'tools/finalize_vot_template_veto_low22.py',
    'tools/finalize_vot_transaction_low22_diagnostics.py',
    'tools/finalize_vot_template_veto_low22_diagnostics.py',
    'tools/smoke_sutrack_template_veto_integration.py',
    'tools/smoke_sutrack_template_veto_parity.py',
    'tools/smoke_sutrack_transaction_gpu.py',
    'tools/diagnose_vot_template_transaction_outcomes.py',
)
base.SOURCE_FILES = tuple(dict.fromkeys(
    tuple(base.common.IMPLEMENTATION_FILES) + base.TRANSACTION_SOURCE_FILES))


if __name__ == '__main__':
    base.main()
