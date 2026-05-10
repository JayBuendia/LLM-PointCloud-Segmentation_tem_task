import argparse
import json
import os

import torch
import torch.nn.functional as F

S3DIS_ORDER = [
    "ceiling", "floor", "wall", "beam", "column", "window", "door",
    "table", "chair", "sofa", "bookcase", "board", "clutter",
]


def encode_with_clip(descriptions, model_name, device):
    try:
        import clip
    except Exception as exc:
        raise RuntimeError(
            "CLIP is not installed. Install with: pip install ftfy regex tqdm && "
            "pip install git+https://github.com/openai/CLIP.git"
        ) from exc

    model, _ = clip.load(model_name, device=device)
    model.eval()
    rows = []
    with torch.no_grad():
        for class_name in S3DIS_ORDER:
            texts = descriptions[class_name]
            tokens = clip.tokenize(texts).to(device)
            feat = model.encode_text(tokens).float()
            feat = F.normalize(feat, dim=-1)
            proto = F.normalize(feat.mean(dim=0), dim=0)
            rows.append(proto.cpu())
    return torch.stack(rows, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptions", default="language_prior/s3dis_descriptions.json")
    parser.add_argument("--output", default="language_prior/s3dis_clip_text_prototypes.pt")
    parser.add_argument("--encoder", default="clip", choices=["clip"])
    parser.add_argument("--model_name", default="ViT-B/32")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    with open(args.descriptions, "r", encoding="utf-8") as f:
        descriptions = json.load(f)
    missing = [c for c in S3DIS_ORDER if c not in descriptions]
    if missing:
        raise ValueError("Missing descriptions for classes: %s" % missing)

    if args.encoder == "clip":
        prototypes = encode_with_clip(descriptions, args.model_name, device)
    else:
        raise NotImplementedError(args.encoder)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(
        {
            "prototypes": prototypes,
            "class_order": S3DIS_ORDER,
            "encoder": args.encoder,
            "model_name": args.model_name,
            "source": os.path.abspath(args.descriptions),
        },
        args.output,
    )
    print("saved", args.output, tuple(prototypes.shape))


if __name__ == "__main__":
    main()
