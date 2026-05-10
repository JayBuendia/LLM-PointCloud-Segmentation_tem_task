# LLM/Text Prototype + Utonia Point Cloud Segmentation

This is a project for reproducing and extending recent LLM-guided point cloud segmentation ideas, including ICLR 2025-style multimodal/few-shot motivation and class-level language priors.

This server workspace is the active project for exploring LLM-guided point cloud segmentation. It reuses the DPA/S3DIS data convention and wraps a Utonia encoder with a semantic segmentation head plus optional CLIP text-prototype guidance.

## Why This Project Exists

The research question is whether language priors from LLM-generated class descriptions can improve point cloud segmentation. The current implementation starts from class-level text prototypes and keeps a path open for stronger point/text or entity/text alignment later.

## Main Files

- `train_utonia_textproto.py`: initial Utonia + text prototype training entry.
- `train_utonia_textproto_opt.py`: optimized training variant used in later runs.
- `train_utonia_textproto_tuned.py`: tuned variant with class-balanced loss, per-class gates, partial freezing, and optimizer groups.
- `models/utonia_backbone.py`: adapter that exposes Utonia features to the segmentation head.
- `language_prior/build_text_prototypes.py`: builds CLIP text prototypes from S3DIS class descriptions.
- `language_prior/s3dis_descriptions.json`: LLM-generated class descriptions.
- `scripts/`: launch scripts for smoke tests, prototype generation, and training variants.
- `PROJECT_STATUS.md`: current experiment summary and next steps.

## Current Limitation

The active server currently reports no GPU. Use this instance for reading, editing, packaging, and log analysis only. Full training should wait for a GPU instance.

## Typical Commands

```bash
# Build CLIP text prototypes when CLIP is installed.
bash scripts/build_clip_text_prototypes.sh

# Quick sanity check on a GPU server.
bash scripts/smoke_baseline.sh

# Baseline and language-guided training variants.
bash scripts/train_baseline.sh
bash scripts/train_textproto.sh
bash scripts/train_textproto_llm_warm_gate.sh
```

## Git Hygiene

The `outputs/` directory is about 14GB and is intentionally ignored. Keep code, configs, launch scripts, language descriptions, and small prototype metadata in Git; keep checkpoints and logs outside Git or upload them as separate artifacts.
