import os
import sys


env_path = os.path.join(os.path.dirname(__file__), '../../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.vot.sutrack_transaction_class import run_vot_exp


run_vot_exp(
    'sutrack_transaction',
    'sutrack_l384_rgbd_anchor_identity_template_transaction_low22',
    vis=False, out_conf=True, channel_type='rgbd')
