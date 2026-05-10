# Project Status

## Current Server State

Path: `/root/autodl-tmp/workspace/llm_pointseg`

The current AutoDL instance has no visible GPU, so this repo is in code/project maintenance mode only. Do not launch full training until a GPU instance is available.

## Implemented Pieces

- S3DIS block loader using raw labels `0..12` and Area 5 validation.
- Pretrained 3D backbone adapter under `models/pretrained_backbone.py`.
- Visual segmentation head plus optional CLIP text-prototype head.
- LLM description file for S3DIS classes.
- CLIP text prototype builder under `language_prior/build_text_prototypes.py`.
- Training scripts for baseline, text prototype, learnable text gate, and tuned full fine-tuning variants.

## Best Runs Observed From Logs

- Full fine-tune flash, lr=3e-6, bs32 from global best: Area 5 val mIoU 0.6388.
- Full fine-tune flash, lr=5e-6, bs8 resume: Area 5 val mIoU 0.6371.
- Tuned proxy freeze class-balanced per-class gate: Area 5 proxy val mIoU 0.6373.
- Text prototype warm gate flash: Area 5 val mIoU around 0.4535.
- Class-name/text prototype only runs are much weaker than the full fine-tuned 3D segmentation baseline, so the language branch should be treated as an auxiliary/gated signal rather than a standalone classifier.

## Important Interpretation

The current strongest results appear to come from full fine-tuning of the 3D segmentation backbone. The LLM/CLIP branch is implemented, but the early logs suggest naive text-logit fusion is not enough. The next meaningful research step is explicit alignment: prototype-text contrast, point-text contrast, or a more careful learnable gate.

## Next GPU Steps

1. Re-run one smoke test after GPU is available.
2. Resume the strongest full fine-tuning checkpoint and validate with consistent Area 5 evaluation.
3. Compare these ablations under the same seed and validation protocol:
   - visual-only pretrained 3D backbone;
   - class-name CLIP prototypes;
   - LLM-description CLIP prototypes;
   - learnable global gate;
   - learnable per-class gate;
   - prototype/text contrastive auxiliary loss.
4. Save a compact result table for paper writing.
