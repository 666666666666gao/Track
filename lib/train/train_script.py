import os
import subprocess
import sys
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.mplt_track import build_mplt_track
# forward propagation related
from lib.train.actors import MPLTTrackActor
# for import modules
import importlib

from ..utils.focal_loss import FocalLoss


def run(settings):
    settings.description = 'Training script for MPLT RGB-T Tracker'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir

    # Create network
    if settings.script_name == "mplt_track":
        net = build_mplt_track(cfg)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    # Loss functions and Actors
    if settings.script_name == "mplt_track":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0}
        actor = MPLTTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        raise ValueError("illegal script name")

    # if cfg.TRAIN.DEEP_SUPERVISION:
    #     raise ValueError("Deep supervision is not supported now.")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # train process
    train_success = trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)

    if train_success and settings.local_rank in [-1, 0] and getattr(cfg.TEST, "AUTO_DEPTHTRACK_EVAL", False):
        eval_epochs = list(getattr(cfg.TEST, "EVAL_EPOCHS", []))
        if not eval_epochs:
            eval_epochs = list(range(max(1, cfg.TRAIN.EPOCH - 5), cfg.TRAIN.EPOCH + 1))
        prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        output_json = os.path.join(
            settings.save_dir, 'test',
            '{}_{}_depthtrack_prre.json'.format(settings.script_name, settings.config_name)
        )
        eval_cmd = [
            sys.executable,
            os.path.join(prj_dir, 'tracking', 'evaluate_depthtrack_prre.py'),
            settings.script_name,
            settings.config_name,
            '--epochs',
        ] + [str(epoch) for epoch in eval_epochs] + [
            '--dataset_name', getattr(cfg.TEST, "EVAL_DATASET", "depthtrack_test"),
            '--checkpoint_root', settings.save_dir,
            '--threads', str(getattr(cfg.TEST, "EVAL_THREADS", 0)),
            '--num_gpus', str(getattr(cfg.TEST, "EVAL_NUM_GPUS", 1)),
            '--output_json', output_json,
        ]
        print('Running DepthTrack Pr/Re/F-score evaluation:')
        print(' '.join(eval_cmd))
        eval_env = os.environ.copy()
        eval_env['MPLT_CHECKPOINT_ROOT'] = settings.save_dir
        result = subprocess.run(eval_cmd, cwd=prj_dir, env=eval_env)
        if result.returncode != 0:
            print('DepthTrack Pr/Re/F-score evaluation failed with return code {}'.format(result.returncode))
