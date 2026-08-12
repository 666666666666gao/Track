# RGB-D Baseline Patch

This patch turns the uploaded RGB-T MPLT/OSTrack-style code into a usable RGB-D baseline.

## What changed

- Added `lib/train/dataset/depthtrack.py` for RGB-D training.
- Added `lib/test/evaluation/depthtrackdataset.py` for RGB-D evaluation.
- Added DepthTrack dataset names to the training and testing registries.
- Added robust depth reading: 16-bit or grayscale depth maps are normalized to 3-channel uint8 pseudo images.
- Added `experiments/mplt_track/vitb_256_mplt_32x1_1e4_depthtrack_15ep_sot.yaml`.

## Expected DepthTrack-style layout

The loader is flexible, but the recommended layout is:

```text
data/depthtrack/train/SEQ_NAME/
  color/00000001.jpg
  depth/00000001.png
  groundtruth.txt

data/depthtrack/test/SEQ_NAME/
  color/00000001.jpg
  depth/00000001.png
  groundtruth.txt
```

The loader also accepts common aliases such as `rgb/`, `visible/`, `img/` for RGB and `depths/`, `depth_colormap/`, `infrared/` for depth.

## Set paths

Edit:

```text
lib/train/admin/local.py
lib/test/evaluation/local.py
```

Set:

```python
self.depthtrack_dir = '/path/to/depthtrack/train'
self.depthtrack_test_dir = '/path/to/depthtrack/test'
settings.depthtrack_path = '/path/to/depthtrack/test'
```

## Train

```bash
python tracking/train.py \
  --script mplt_track \
  --config vitb_256_mplt_32x1_1e4_depthtrack_15ep_sot \
  --save_dir ./output/vitb_256_mplt_32x1_1e4_depthtrack_15ep_sot \
  --mode multiple \
  --nproc_per_node 4
```

For single GPU:

```bash
python tracking/train.py \
  --script mplt_track \
  --config vitb_256_mplt_32x1_1e4_depthtrack_15ep_sot \
  --save_dir ./output/vitb_256_mplt_32x1_1e4_depthtrack_15ep_sot \
  --mode single \
  --nproc_per_node 1
```

## Test

```bash
python tracking/test.py \
  mplt_track \
  vitb_256_mplt_32x1_1e4_depthtrack_15ep_sot \
  --dataset_name depthtrack_test \
  --threads 6 \
  --num_gpus 1
```

## Baseline note

This is not yet the full RGB-D-L method. It is the RGB-D baseline branch. Your language decomposition, target-context matching, and depth-reliable routing modules can be added on top of this baseline.
