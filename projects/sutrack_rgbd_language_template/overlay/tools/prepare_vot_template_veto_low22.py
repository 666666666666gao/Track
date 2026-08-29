#!/usr/bin/env python3
"""Prepare the frozen low22 workspace for baseline-first template veto."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import prepare_vot_transaction_low22 as base


base.TRACKER = 'sutrack_l384_rgbd_anchor_identity_template_veto_low22'


if __name__ == '__main__':
    base.main()
