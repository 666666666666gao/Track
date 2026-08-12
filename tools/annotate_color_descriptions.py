#!/usr/bin/env python3
"""Create frame-aligned RGB-D color descriptions with local Qwen models.

The output deliberately follows the existing plain-text annotation convention:

    color_desc/<dataset>/<sequence>/color_description_qwen25.txt
    color_desc/<dataset>/<sequence>/color_description_ct.txt

For each supported evaluation sequence, both files contain exactly one English
sentence for its initial RGB frame. The first file is the resumable Qwen2.5-VL
draft; the second is the Qwen3-corrected result consumed by the tracker. This
tool never writes a JSONL annotation schema. Existing DepthTrack-train files
remain frame-aligned with one sentence per RGB frame and are not rewritten.
"""

from __future__ import print_function

import argparse
import gc
import hashlib
import json
import math
import os
import re
import sys
import tempfile


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_ROOTS = {
    "depthtrack_test": r"D:\steam\RGB-D-L-work\datasets\DepthTrackTest",
    "cdtb": r"D:\steam\RGB-D-L-work\datasets\CDTB",
    "votrgbd2022": r"D:\steam\RGB-D-L-work\datasets\VOT-RGBD2022",
}
DEFAULT_QWEN25_MODEL = r"D:\steam\RGB-D-L-work\models\Qwen_VL\qwen2.5-vl-7b"
DEFAULT_QWEN3_MODEL = r"D:\steam\RGB-D-L-work\models\Qwen3_8B"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
RGB_DIR_CANDIDATES = ("color", "rgb", "visible", "img", "imgs", "image", "images")
GT_FILE_CANDIDATES = ("groundtruth.txt", "groundtruth_rect.txt", "init.txt", "rgb.txt")
RAW_FILENAME = "color_description_qwen25.txt"
CORRECTED_FILENAME = "color_description_ct.txt"
FORBIDDEN_TERMS = (
    "bounding box",
    "bbox",
    "coordinate",
    "coordinates",
    "marked region",
    "marked area",
    "marked object",
    "selected region",
    "<box>",
    "</box>",
)
RESUME_SCHEMA_VERSION = 1
ANNOTATION_SCOPE = "initial_frame"
QWEN25_PROMPT_VERSION = "qwen25-initial-frame-v1"
QWEN3_PROMPT_VERSION = "qwen3-text-correction-v5"


def _atomic_write_bytes(path, content):
    directory = os.path.dirname(path) or os.curdir
    if not os.path.isdir(directory):
        os.makedirs(directory)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=directory, prefix=".{}-".format(os.path.basename(path)), suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.remove(temporary_path)
        except OSError:
            pass
        raise


class DescriptionCheckpoint(object):
    """Crash-safe writer for a plain-text description file and its sidecar state."""

    def __init__(self, path, metadata, overwrite=False, create=True):
        self.path = os.path.abspath(path)
        self.state_path = self.path + ".resume.json"
        self.metadata = dict(metadata)
        self.lines = []
        self._committed_bytes = 0

        output_exists = os.path.isfile(self.path)
        state_exists = os.path.isfile(self.state_path)
        if overwrite:
            self._initialize()
            return
        if not output_exists and not state_exists:
            if not create:
                raise FileNotFoundError("Description checkpoint does not exist: {}".format(self.path))
            self._initialize()
            return
        if not output_exists and state_exists:
            state = self._load_state()
            if state.get("committed_lines") != 0 or state.get("committed_bytes") != 0:
                raise ValueError(
                    "Description text is missing for a non-empty checkpoint: {}".format(self.path))
            _atomic_write_bytes(self.path, b"")
            self._resume()
            return
        if not output_exists or not state_exists:
            raise ValueError(
                "Incomplete description checkpoint for {}. Use --overwrite to restart it.".format(
                    self.path))
        self._resume()

    def _state(self):
        return {
            "schema_version": RESUME_SCHEMA_VERSION,
            "metadata": self.metadata,
            "committed_lines": len(self.lines),
            "committed_bytes": self._committed_bytes,
        }

    def _write_state(self):
        encoded = (json.dumps(self._state(), ensure_ascii=True, sort_keys=True) + "\n").encode("utf-8")
        _atomic_write_bytes(self.state_path, encoded)

    def _initialize(self):
        self.lines = []
        self._committed_bytes = 0
        self._write_state()
        _atomic_write_bytes(self.path, b"")

    def _load_state(self):
        with open(self.state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("schema_version") != RESUME_SCHEMA_VERSION:
            raise ValueError("Unsupported description checkpoint schema: {}".format(self.state_path))
        if state.get("metadata") != self.metadata:
            raise ValueError("Description checkpoint metadata mismatch: {}".format(self.path))
        return state

    def _resume(self):
        state = self._load_state()
        committed_lines = state.get("committed_lines")
        committed_bytes = state.get("committed_bytes")
        if not isinstance(committed_lines, int) or committed_lines < 0:
            raise ValueError("Invalid committed line count in {}".format(self.state_path))
        if not isinstance(committed_bytes, int) or committed_bytes < 0:
            raise ValueError("Invalid committed byte count in {}".format(self.state_path))

        with open(self.path, "rb") as handle:
            content = handle.read()
        if len(content) < committed_bytes:
            raise ValueError("Description file is shorter than its checkpoint: {}".format(self.path))
        committed_content = content[:committed_bytes]
        try:
            committed_text = committed_content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Committed description text is not valid UTF-8: {}".format(self.path)) from exc
        lines = committed_text.splitlines()
        if len(lines) != committed_lines or any(not line.strip() for line in lines):
            raise ValueError("Committed description lines are invalid: {}".format(self.path))
        if committed_content and not committed_content.endswith(b"\n"):
            raise ValueError("Committed description text has no line terminator: {}".format(self.path))

        self.lines = lines
        self._committed_bytes = committed_bytes
        if len(content) != committed_bytes:
            _atomic_write_bytes(self.path, committed_content)

    def append(self, description):
        description = str(description or "").strip()
        if not description:
            raise ValueError("Description is empty")
        if "\n" in description or "\r" in description:
            raise ValueError("Description must contain exactly one plain-text line")
        encoded = (description + "\n").encode("utf-8")
        with open(self.path, "ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        self.lines.append(description)
        self._committed_bytes += len(encoded)
        self._write_state()


def natural_sort_key(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = re.split(r"(\d+)", stem.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def _find_directory(sequence_dir, candidates):
    for name in candidates:
        path = os.path.join(sequence_dir, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        "Cannot find an RGB directory under {}. Tried {}".format(sequence_dir, candidates))


def _find_ground_truth(sequence_dir):
    for name in GT_FILE_CANDIDATES:
        path = os.path.join(sequence_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("Cannot find ground truth under {}".format(sequence_dir))


def _parse_box(values, path, line_number):
    if len(values) >= 8:
        xs = values[0::2]
        ys = values[1::2]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    if len(values) >= 4:
        return tuple(values[:4])
    raise ValueError(
        "Expected four rectangle values or eight polygon values in {}:{}".format(path, line_number))


def read_ground_truth(path):
    boxes = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                values = [float(value) for value in re.split(r"[\s,\t]+", line) if value]
            except ValueError:
                raise ValueError("Non-numeric ground truth in {}:{}".format(path, line_number))
            boxes.append(_parse_box(values, path, line_number))
    if not boxes:
        raise ValueError("Ground-truth file is empty: {}".format(path))
    return boxes


def list_rgb_frames(rgb_dir):
    frames = [
        os.path.join(rgb_dir, name)
        for name in os.listdir(rgb_dir)
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    ]
    frames.sort(key=natural_sort_key)
    if not frames:
        raise ValueError("No RGB frames found in {}".format(rgb_dir))
    return frames


def _validate_box(box, sequence_dir, frame_index):
    if len(box) != 4 or not all(math.isfinite(value) for value in box):
        raise ValueError(
            "Ground truth has invalid values at frame {} in {}".format(frame_index, sequence_dir))
    if box[2] <= 0 or box[3] <= 0:
        raise ValueError(
            "Ground truth has invalid size at frame {} in {}".format(frame_index, sequence_dir))


def collect_sequence_inputs(sequence_dir, required_frame_ids=None):
    """Return naturally sorted RGB frame paths and exactly aligned boxes."""
    rgb_dir = _find_directory(sequence_dir, RGB_DIR_CANDIDATES)
    frames = list_rgb_frames(rgb_dir)
    boxes = read_ground_truth(_find_ground_truth(sequence_dir))
    if len(frames) != len(boxes):
        raise ValueError(
            "Frame/ground-truth count mismatch for {}: {} RGB frames, {} boxes".format(
                sequence_dir, len(frames), len(boxes)))
    frame_ids = range(len(boxes)) if required_frame_ids is None else required_frame_ids
    for frame_index in frame_ids:
        if frame_index < 0 or frame_index >= len(boxes):
            raise IndexError(
                "Required frame {} is outside {} with {} frames".format(
                    frame_index, sequence_dir, len(boxes)))
        _validate_box(boxes[frame_index], sequence_dir, frame_index)
    return frames, boxes


def discover_sequences(dataset_root):
    data_root = os.path.join(dataset_root, "sequences")
    if not os.path.isdir(data_root):
        data_root = dataset_root
    if not os.path.isdir(data_root):
        raise FileNotFoundError("Dataset directory does not exist: {}".format(dataset_root))

    sequences = []
    for name in sorted(os.listdir(data_root)):
        sequence_dir = os.path.join(data_root, name)
        if not os.path.isdir(sequence_dir):
            continue
        try:
            _find_ground_truth(sequence_dir)
            _find_directory(sequence_dir, RGB_DIR_CANDIDATES)
        except FileNotFoundError:
            continue
        sequences.append((name, sequence_dir))
    if not sequences:
        raise ValueError("No RGB-D tracking sequences found under {}".format(data_root))
    return sequences


def _clean_text(text):
    text = str(text or "")
    text = re.sub(r"```(?:text|json)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:description|corrected description)\s*:\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n\"'")
    return text


def _is_valid_corrected_text(text):
    text = _clean_text(text)
    if len(text.split()) < 3 or len(text.split()) > 20:
        return False
    low = text.lower()
    return not any(term in low for term in FORBIDDEN_TERMS)


def _correction_retry_feedback(previous_attempt, retry_number=1):
    cleaned = _clean_text(previous_attempt)
    word_count = len(cleaned.split())
    present_terms = [term for term in FORBIDDEN_TERMS if term in cleaned.lower()]
    issues = ["the previous answer has {} words".format(word_count)]
    if word_count > 20:
        issues.append("remove location and background details")
    if word_count < 3:
        issues.append("include a concise target description")
    if present_terms:
        issues.append("remove forbidden wording: {}".format(", ".join(present_terms)))
    if retry_number <= 1:
        target_length = "12 to 18 words"
        focus = "Remove unnecessary details."
    elif retry_number == 2:
        target_length = "8 to 12 words"
        focus = "Keep only the object category, color, shape, and material; remove people, scene details, and guesses."
    else:
        target_length = "6 to 10 words"
        focus = "Use a minimal subject-description sentence with only category, color, shape, or material."
    return (
        "Retry {retry}: {issues}; rewrite it in {target_length}. {focus} "
        "Do not copy the previous sentence."
    ).format(
        retry=int(retry_number), issues="; ".join(issues),
        target_length=target_length, focus=focus)


def _category_hint(sequence_name):
    return sequence_name.split("_", 1)[0].replace("-", " ")


def _clip_box(box, width, height):
    try:
        x, y, w, h = [float(value) for value in box]
    except (TypeError, ValueError):
        raise ValueError("Ground-truth box must contain four numeric values")
    if not all(math.isfinite(value) for value in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError("Ground-truth box must be finite with positive width and height")
    left = max(0.0, x)
    top = max(0.0, y)
    right = min(float(width), x + w)
    bottom = min(float(height), y + h)
    if right <= left or bottom <= top:
        raise ValueError("Ground-truth box does not intersect the RGB frame")
    return (left, top, right - left, bottom - top)


class Qwen25Captioner(object):
    """Lazy Qwen2.5-VL runner for visual draft descriptions."""

    def __init__(self, model_path, gpu_memory, cpu_memory, load_in_4bit=False):
        _validate_memory_limits(load_in_4bit, gpu_memory, cpu_memory)
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError(
                "Qwen2.5-VL requires torch, transformers, and qwen_vl_utils in the selected Python environment") from exc

        if not os.path.isdir(model_path):
            raise FileNotFoundError("Qwen2.5-VL model directory does not exist: {}".format(model_path))
        self.torch = torch
        self.process_vision_info = process_vision_info
        model_kwargs = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto",
        }
        max_memory = {}
        if torch.cuda.is_available() and gpu_memory:
            max_memory[0] = gpu_memory
        if cpu_memory:
            max_memory["cpu"] = cpu_memory
        if max_memory:
            model_kwargs["max_memory"] = max_memory
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                    llm_int8_skip_modules=["visual", "lm_head"],
                )
                # Keep only quantized language Transformer blocks on the small
                # GPU.  The unquantized vision tower and vocabulary matrices
                # remain on CPU, avoiding unsupported 4-bit CPU execution.
                model_kwargs["device_map"] = {
                    "visual": "cpu",
                    "model.embed_tokens": "cpu",
                    "model.layers": 0,
                    "model.norm": 0,
                    "model.rotary_emb": 0,
                    "lm_head": "cpu",
                }
                model_kwargs.pop("max_memory", None)
            except ImportError as exc:
                raise RuntimeError("--load-in-4bit requires bitsandbytes") from exc
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        self.model.eval()

    def _input_device(self):
        device_map = getattr(self.model, "hf_device_map", {})
        mapped_device = device_map.get("model.embed_tokens")
        if mapped_device is not None:
            if isinstance(mapped_device, int):
                return self.torch.device("cuda:{}".format(mapped_device))
            return self.torch.device(mapped_device)
        device = next(self.model.parameters()).device
        if getattr(device, "type", "") == "meta":
            return self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        return device

    def describe(self, image_path, box, sequence_name):
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        x, y, w, h = _clip_box(box, image.width, image.height)
        category = _category_hint(sequence_name)
        prompt = (
            "Describe the tracked {category} in the marked region of this RGB frame. "
            "Focus on visible color, shape, material, pose or motion, and details that distinguish it from nearby objects. "
            "Return one factual English sentence under 20 words. "
            "Do not mention markings, boxes, coordinates, or explain your reasoning. "
            "The region is <box>({x:.1f},{y:.1f},{right:.1f},{bottom:.1f})</box>."
        ).format(category=category, x=x, y=y, right=x + w, bottom=y + h)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(self._input_device())
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=48,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        generated = generated[:, inputs.input_ids.shape[1]:]
        return _clean_text(self.processor.batch_decode(generated, skip_special_tokens=True)[0])

    def close(self):
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class Qwen3Corrector(object):
    """Lazy Qwen3 runner that corrects text without inventing visual details."""

    def __init__(self, model_path, gpu_memory, cpu_memory, load_in_4bit=False):
        _validate_memory_limits(load_in_4bit, gpu_memory, cpu_memory)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Qwen3 requires torch and transformers in the selected Python environment") from exc

        if not os.path.isdir(model_path):
            raise FileNotFoundError("Qwen3 model directory does not exist: {}".format(model_path))
        self.torch = torch
        model_kwargs = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto",
        }
        max_memory = {}
        if torch.cuda.is_available() and gpu_memory:
            max_memory[0] = gpu_memory
        if cpu_memory:
            max_memory["cpu"] = cpu_memory
        if max_memory:
            model_kwargs["max_memory"] = max_memory
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                    llm_int8_skip_modules=["lm_head"],
                )
                model_kwargs["device_map"] = {
                    "model.embed_tokens": "cpu",
                    "model.layers": 0,
                    "model.norm": 0,
                    "model.rotary_emb": 0,
                    "lm_head": "cpu",
                }
                model_kwargs.pop("max_memory", None)
            except ImportError as exc:
                raise RuntimeError("--load-in-4bit requires bitsandbytes") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self.model.eval()

    def _input_device(self):
        device_map = getattr(self.model, "hf_device_map", {})
        mapped_device = device_map.get("model.embed_tokens")
        if mapped_device is not None:
            if isinstance(mapped_device, int):
                return self.torch.device("cuda:{}".format(mapped_device))
            return self.torch.device(mapped_device)
        device = next(self.model.parameters()).device
        if getattr(device, "type", "") == "meta":
            return self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        return device

    def correct(self, draft, previous_attempt=None, retry_number=0):
        prompt = (
            "Rewrite this tracking-object description as one factual English sentence under 20 words. "
            "Preserve only details already present. Do not add visual details. "
            "Do not mention boxes, coordinates, annotations, prompts, or reasoning. "
            "Output only the corrected sentence.\n\nDraft: {}"
        ).format(draft)
        if previous_attempt is not None:
            prompt += (
                "\n\nYour previous answer was invalid: {}"
                "\nTry again. Use the Draft as the only factual source and output one valid sentence."
            ).format(_correction_retry_feedback(previous_attempt, retry_number))
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self._input_device())
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=48,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        output = generated[0][inputs.input_ids.shape[1]:]
        return _clean_text(self.tokenizer.decode(output, skip_special_tokens=True))

    def close(self):
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def _stage_path(output_root, dataset_name, sequence_name, filename):
    return os.path.join(output_root, dataset_name, sequence_name, filename)


def _sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _initial_frame_signature(frames, boxes):
    if not frames or not boxes:
        raise ValueError("An initial RGB frame and ground-truth box are required")
    payload = {
        "frame_name": os.path.basename(frames[0]),
        "frame_sha256": _sha256_file(frames[0]),
        "ground_truth": [float(value) for value in boxes[0]],
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(encoded.encode("utf-8"))


def _stage_metadata(args, dataset_name, sequence_name, frames, boxes, stage, draft=None):
    metadata = {
        "annotation_scope": ANNOTATION_SCOPE,
        "dataset": dataset_name,
        "sequence": sequence_name,
        "source_sha256": _initial_frame_signature(frames, boxes),
        "plain_text_filename": RAW_FILENAME if stage == "qwen25" else CORRECTED_FILENAME,
    }
    if stage == "qwen25":
        metadata.update({
            "model_path": os.path.abspath(args.qwen25_model),
            "prompt_version": QWEN25_PROMPT_VERSION,
        })
    elif stage == "qwen3":
        if draft is None:
            raise ValueError("Qwen3 checkpoint metadata requires the Qwen2.5 draft")
        metadata.update({
            "draft_sha256": _sha256_bytes(draft.encode("utf-8")),
            "qwen25_model_path": os.path.abspath(args.qwen25_model),
            "model_path": os.path.abspath(args.qwen3_model),
            "prompt_version": QWEN3_PROMPT_VERSION,
        })
    else:
        raise ValueError("Unsupported annotation stage: {}".format(stage))
    return metadata


def _selected_sequences(dataset_root, max_sequences):
    sequences = discover_sequences(dataset_root)
    return sequences[:max_sequences] if max_sequences else sequences


def _frame_limit(frame_count, max_frames_per_sequence):
    # The supported datasets are evaluation sets. Their official annotation
    # contract is one initial-frame sentence per sequence.
    return min(frame_count, 1)


def _caption_dataset(args, dataset_name, dataset_root):
    sequences = _selected_sequences(dataset_root, args.max_sequences)
    if args.dry_run:
        total = 0
        for sequence_name, sequence_dir in sequences:
            frames, _ = collect_sequence_inputs(sequence_dir, required_frame_ids=[0])
            total += _frame_limit(len(frames), args.max_frames_per_sequence)
        print("[dry-run] {}: {} sequences, {} Qwen2.5 frames".format(dataset_name, len(sequences), total))
        return

    captioner = Qwen25Captioner(
        args.qwen25_model, args.gpu_memory, args.cpu_memory, args.load_in_4bit)
    try:
        for sequence_number, (sequence_name, sequence_dir) in enumerate(sequences, 1):
            frames, boxes = collect_sequence_inputs(sequence_dir, required_frame_ids=[0])
            limit = _frame_limit(len(frames), args.max_frames_per_sequence)
            output = _stage_path(args.output_root, dataset_name, sequence_name, RAW_FILENAME)
            metadata = _stage_metadata(
                args, dataset_name, sequence_name, frames, boxes, stage="qwen25")
            checkpoint = DescriptionCheckpoint(output, metadata, overwrite=args.overwrite)
            existing = checkpoint.lines
            if len(existing) > limit:
                raise ValueError("Existing draft has too many lines: {}".format(output))
            print("Qwen2.5 {} {}/{}: {}/{} frames".format(
                dataset_name, sequence_number, len(sequences), len(existing), limit))
            for frame_index in range(len(existing), limit):
                description = captioner.describe(frames[frame_index], boxes[frame_index], sequence_name)
                if not description:
                    raise RuntimeError("Qwen2.5 returned an empty description for {} frame {}".format(
                        sequence_name, frame_index))
                checkpoint.append(description)
    finally:
        captioner.close()


def _correct_dataset(args, dataset_name, dataset_root):
    sequences = _selected_sequences(dataset_root, args.max_sequences)
    if args.dry_run:
        total = 0
        for sequence_name, sequence_dir in sequences:
            frames, _ = collect_sequence_inputs(sequence_dir, required_frame_ids=[0])
            limit = _frame_limit(len(frames), args.max_frames_per_sequence)
            total += limit
        print("[dry-run] {}: {} sequences, {} Qwen3 corrections".format(dataset_name, len(sequences), total))
        return

    corrector = Qwen3Corrector(args.qwen3_model, args.gpu_memory, args.cpu_memory, args.load_in_4bit)
    try:
        for sequence_number, (sequence_name, sequence_dir) in enumerate(sequences, 1):
            frames, boxes = collect_sequence_inputs(sequence_dir, required_frame_ids=[0])
            limit = _frame_limit(len(frames), args.max_frames_per_sequence)
            raw_path = _stage_path(args.output_root, dataset_name, sequence_name, RAW_FILENAME)
            raw_metadata = _stage_metadata(
                args, dataset_name, sequence_name, frames, boxes, stage="qwen25")
            raw_checkpoint = DescriptionCheckpoint(
                raw_path, raw_metadata, create=False)
            raw_lines = raw_checkpoint.lines
            if len(raw_lines) != limit:
                raise ValueError(
                    "Expected exactly one Qwen2.5 draft line for {}: {}".format(
                        sequence_name, raw_path))
            output = _stage_path(args.output_root, dataset_name, sequence_name, CORRECTED_FILENAME)
            metadata = _stage_metadata(
                args, dataset_name, sequence_name, frames, boxes,
                stage="qwen3", draft=raw_lines[0])
            checkpoint = DescriptionCheckpoint(output, metadata, overwrite=args.overwrite)
            existing = checkpoint.lines
            if len(existing) > limit:
                raise ValueError("Existing corrected file has too many lines: {}".format(output))
            print("Qwen3 {} {}/{}: {}/{} frames".format(
                dataset_name, sequence_number, len(sequences), len(existing), limit))
            for frame_index in range(len(existing), limit):
                corrected = ""
                previous_attempt = None
                for retry_number in range(args.max_retries):
                    corrected = corrector.correct(
                        raw_lines[frame_index], previous_attempt=previous_attempt,
                        retry_number=retry_number)
                    if _is_valid_corrected_text(corrected):
                        break
                    previous_attempt = corrected
                if not _is_valid_corrected_text(corrected):
                    raise RuntimeError(
                        "Qwen3 could not produce a valid corrected description for {} frame {}".format(
                            sequence_name, frame_index))
                checkpoint.append(corrected)
    finally:
        corrector.close()


def _parse_data_root_overrides(items):
    roots = dict(DEFAULT_DATA_ROOTS)
    for item in items or []:
        if "=" not in item:
            raise ValueError("--data-root must use dataset=path, got {}".format(item))
        dataset_name, path = item.split("=", 1)
        if dataset_name not in roots:
            raise ValueError("Unknown dataset in --data-root: {}".format(dataset_name))
        roots[dataset_name] = path
    return roots


def _validate_memory_limits(load_in_4bit, gpu_memory, cpu_memory):
    if load_in_4bit and gpu_memory:
        raise ValueError(
            "--gpu-memory is not supported with --load-in-4bit because the quantized device map is fixed")
    if load_in_4bit and cpu_memory:
        raise ValueError(
            "--cpu-memory is not supported with --load-in-4bit because the quantized device map is fixed")


def validate_runtime_options(args):
    _validate_memory_limits(args.load_in_4bit, args.gpu_memory, args.cpu_memory)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=list(DEFAULT_DATA_ROOTS) + ["all"],
        default=["all"],
        help="Datasets to annotate. Existing DepthTrack train color_desc is left untouched.",
    )
    parser.add_argument(
        "--data-root",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Override one source dataset root.",
    )
    parser.add_argument("--output-root", default=os.path.join(PROJECT_ROOT, "color_desc"))
    parser.add_argument("--qwen25-model", default=DEFAULT_QWEN25_MODEL)
    parser.add_argument("--qwen3-model", default=DEFAULT_QWEN3_MODEL)
    parser.add_argument("--stage", choices=("qwen25", "qwen3", "all"), default="all")
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--gpu-memory", default=None,
        help="Optional Accelerate GPU cap for non-quantized loading; unsupported with --load-in-4bit.")
    parser.add_argument(
        "--cpu-memory", default=None,
        help="Optional Accelerate CPU cap for non-quantized loading; unsupported with --load-in-4bit.")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    validate_runtime_options(args)
    if args.max_sequences is not None and args.max_sequences <= 0:
        raise ValueError("--max-sequences must be positive")
    if args.max_frames_per_sequence is not None and args.max_frames_per_sequence <= 0:
        raise ValueError("--max-frames-per-sequence must be positive")
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be positive")

    roots = _parse_data_root_overrides(args.data_root)
    datasets = list(roots) if "all" in args.dataset else args.dataset
    args.output_root = os.path.abspath(args.output_root)
    print("Output root: {}".format(args.output_root))
    for dataset_name in datasets:
        dataset_root = roots[dataset_name]
        if args.stage in ("qwen25", "all"):
            _caption_dataset(args, dataset_name, dataset_root)
        if args.stage in ("qwen3", "all"):
            _correct_dataset(args, dataset_name, dataset_root)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        raise
