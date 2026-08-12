# RGB-D / RGB-D-L Dataset Support

This package is adapted for your generated RGB-D language annotations and three evaluation/training sources:

- DepthTrack train/test
- CDTB evaluation
- VOT-RGBD2022 evaluation

## Expected data layout

The loaders support either of these layouts:

```text
data/depthtrack/train/<sequence>/color/00000001.jpg
data/depthtrack/train/<sequence>/depth/00000001.png
data/depthtrack/train/<sequence>/groundtruth.txt

# or

data/CDTB/sequences/<sequence>/color/00000001.jpg
data/CDTB/sequences/<sequence>/depth/00000001.png
data/CDTB/sequences/<sequence>/groundtruth.txt
```

Common aliases are also supported: `rgb`, `visible`, `img`, `images` for RGB and `depth`, `depths`, `depth_colormap`, `depth_color` for depth.

## Included annotation files

The cleaned annotations are under `annotations/`:

```text
annotations/depthtrack_train_first_qwen3_corrected.jsonl
annotations/depthtrack_test_first_qwen3_corrected.jsonl
annotations/cdtb_first_qwen3_corrected.jsonl
annotations/vot_rgbd2022_first_qwen3_corrected.jsonl
annotations/CDTBLang_Qwen3_RAGStyle/
annotations/VOTRGBD2022Lang_Qwen3_RAGStyle/
```

The loader keys annotations by sequence name. It does not depend on the absolute Windows paths saved inside the JSONL metadata.

## What was fixed in the annotations

- `similar_depth` was normalized to `similar_to_background`.
- `poor` depth quality was normalized to `low`.
- Phrases like `bounding box` and `red bounding box` were removed from target descriptions.
- `depth_stats` labels were normalized to match the final annotation labels.

Run this to check annotations:

```bash
python tools/check_rgbd_language_annotations.py annotations/depthtrack_train_first_qwen3_corrected.jsonl
python tools/check_rgbd_language_annotations.py annotations/CDTBLang_Qwen3_RAGStyle
```

## Path settings

Edit these paths if your datasets are elsewhere:

```python
# lib/train/admin/local.py
self.depthtrack_dir = './data/depthtrack/train'
self.depthtrack_test_dir = './data/depthtrack/test'
self.depthtrack_train_lang_path = './annotations/depthtrack_train_first_qwen3_corrected.jsonl'
self.depthtrack_test_lang_path = './annotations/depthtrack_test_first_qwen3_corrected.jsonl'

# lib/test/evaluation/local.py
settings.depthtrack_path = './data/depthtrack/test'
settings.cdtb_path = './data/CDTB'
settings.votrgbd2022_path = './data/VOTRGBD2022'
```

## Training DepthTrack RGB-D baseline

```bash
python tracking/train.py \
  --script mplt_track \
  --config vitb_256_mplt_32x1_1e4_depthtrack_15ep_sot \
  --save_dir ./output/depthtrack_rgbd_baseline \
  --mode multiple \
  --nproc_per_node 4
```

The current RGB-D baseline does not consume language features in the network. However, the sampler now passes the following fields, so your RGB-D-L method can use them directly:

```text
language_description
language_appearance
language_depth_relation
language_depth_quality
language_occlusion_state
language_distractor_relation
```

## Testing

```bash
python tracking/test.py mplt_track vitb_256_mplt_32x1_1e4_depthtrack_15ep_sot --dataset_name depthtrack_test --threads 6 --num_gpus 1
python tracking/test.py mplt_track vitb_256_mplt_32x1_1e4_depthtrack_15ep_sot --dataset_name cdtb --threads 6 --num_gpus 1
python tracking/test.py mplt_track vitb_256_mplt_32x1_1e4_depthtrack_15ep_sot --dataset_name votrgbd2022 --threads 6 --num_gpus 1
```
