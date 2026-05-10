import argparse
import ast
import glob
import math
import os
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

S3DIS_ORDER = [
    "ceiling", "floor", "wall", "beam", "column", "window", "door",
    "table", "chair", "sofa", "bookcase", "board", "clutter",
]


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def augment_xyz(xyz, args):
    if args.aug_scale > 1:
        scale = random.uniform(1.0 / args.aug_scale, args.aug_scale)
        xyz = xyz * scale
    if args.aug_rot:
        theta = random.uniform(0.0, 2.0 * math.pi)
        c, s = math.cos(theta), math.sin(theta)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        xyz = xyz @ rot.T
    if args.aug_mirror_prob > 0:
        if random.random() < args.aug_mirror_prob * 0.5:
            xyz[:, 0] *= -1
        if random.random() < args.aug_mirror_prob * 0.5:
            xyz[:, 1] *= -1
    if args.aug_jitter:
        xyz = xyz + np.clip(0.01 * np.random.randn(*xyz.shape), -0.05, 0.05).astype(np.float32)
    return xyz


class S3DISBlockDataset(Dataset):
    def __init__(self, data_root, split, test_area="Area_5", num_points=2048, pc_attribs="xyzrgbXYZ", augment=False, args=None):
        self.data_root = data_root
        self.data_dir = os.path.join(data_root, "data")
        self.split = split
        self.test_area = test_area
        self.num_points = int(num_points)
        self.pc_attribs = pc_attribs
        self.augment = bool(augment)
        self.args = args
        all_files = sorted(glob.glob(os.path.join(self.data_dir, "*.npy")))
        if not all_files:
            raise FileNotFoundError("No .npy blocks under %s" % self.data_dir)
        if split == "train":
            self.files = [p for p in all_files if not os.path.basename(p).startswith(test_area + "_")]
        elif split in ("val", "test"):
            self.files = [p for p in all_files if os.path.basename(p).startswith(test_area + "_")]
        else:
            raise ValueError("split must be train/val/test")
        if not self.files:
            raise RuntimeError("No files for split=%s test_area=%s" % (split, test_area))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = np.load(self.files[index])
        n_points = data.shape[0]
        choice = np.random.choice(np.arange(n_points), self.num_points, replace=(n_points < self.num_points))
        data = data[choice]
        xyz = data[:, 0:3].astype(np.float32)
        rgb = data[:, 3:6].astype(np.float32)
        label = data[:, 6].astype(np.int64)

        xyz = xyz - np.amin(xyz, axis=0, keepdims=True)
        if self.augment and self.args is not None:
            xyz = augment_xyz(xyz, self.args)

        features = []
        if "xyz" in self.pc_attribs:
            features.append(xyz)
        if "rgb" in self.pc_attribs:
            features.append(rgb / 255.0)
        if "XYZ" in self.pc_attribs:
            xyz_norm = xyz - np.amin(xyz, axis=0, keepdims=True)
            denom = np.amax(xyz_norm, axis=0, keepdims=True)
            xyz_norm = xyz_norm / np.maximum(denom, 1e-6)
            features.append(xyz_norm.astype(np.float32))
        point = np.concatenate(features, axis=1).astype(np.float32)
        return torch.from_numpy(point.T), torch.from_numpy(label)


class TextPrototypeHead(nn.Module):
    def __init__(self, feat_dim, text_dim, init_logit_scale=10.0):
        super().__init__()
        self.point_proj = nn.Conv1d(feat_dim, text_dim, 1, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(float(init_logit_scale)))

    def forward(self, point_features, text_prototypes):
        point_emb = self.point_proj(point_features).transpose(1, 2).contiguous()
        point_emb = F.normalize(point_emb, dim=-1)
        text_prototypes = F.normalize(text_prototypes, dim=-1)
        logits = self.logit_scale.clamp(1.0, 100.0) * torch.matmul(point_emb, text_prototypes.t())
        return logits.transpose(1, 2).contiguous()


class UtoniaSeg(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        dpa_root = args.dpa_root
        if dpa_root not in sys.path:
            sys.path.insert(0, dpa_root)
        from models.utonia_backbone import UtoniaBackbone

        self.freeze_utonia_encoder = bool(args.utonia_freeze_encoder)
        self.encoder = UtoniaBackbone(args)
        self.feat_dim = int(args.utonia_feat_dim) + sum(int(w[-1]) for w in args.edgeconv_widths)
        self.classifier = nn.Sequential(
            nn.Conv1d(self.feat_dim, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(128, num_classes, 1),
        )

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_utonia_encoder and hasattr(self.encoder, "utonia_model"):
            self.encoder.utonia_model.eval()
        return self

    def forward(self, pc, return_features=False):
        edge_feats, point_feat, _ = self.encoder(pc)
        feat = torch.cat(list(edge_feats) + [point_feat], dim=1)
        logits = self.classifier(feat)
        if return_features:
            return logits, feat
        return logits


class TextProtoSeg(nn.Module):
    def __init__(self, args, num_classes, text_prototypes=None):
        super().__init__()
        self.segmentor = UtoniaSeg(args, num_classes)
        self.text_weight = float(args.text_weight)
        self.learnable_text_gate = bool(getattr(args, "learnable_text_gate", False))
        self.text_head = None
        self.text_gate = None
        if text_prototypes is not None:
            self.register_buffer("text_prototypes", text_prototypes.float())
            self.text_head = TextPrototypeHead(self.segmentor.feat_dim, text_prototypes.shape[1], args.text_logit_scale)
            if self.learnable_text_gate:
                init = float(getattr(args, "text_gate_init", 0.02))
                init = min(max(init, 1e-4), 1.0 - 1e-4)
                if bool(getattr(args, "per_class_text_gate", False)):
                    self.text_gate = nn.Parameter(torch.full((num_classes,), torch.logit(torch.tensor(init)).item()))
                else:
                    self.text_gate = nn.Parameter(torch.logit(torch.tensor(init)))
        else:
            self.text_prototypes = None

    def forward(self, pc):
        z_vis, feat = self.segmentor(pc, return_features=True)
        z_text = None
        if self.text_head is not None and (self.text_weight > 0 or self.learnable_text_gate):
            z_text = self.text_head(feat, self.text_prototypes)
            gate = torch.sigmoid(self.text_gate) if self.learnable_text_gate else self.text_weight
            if torch.is_tensor(gate) and gate.ndim == 1:
                gate = gate.view(1, -1, 1)
            return z_vis + gate * z_text, z_vis, z_text
        return z_vis, z_vis, z_text


@dataclass
class Meter:
    total_loss: float = 0.0
    count: int = 0

    def update(self, value, n=1):
        self.total_loss += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self):
        return self.total_loss / max(self.count, 1)


def update_confusion(confusion, pred, target, num_classes):
    pred = pred.reshape(-1).detach().cpu()
    target = target.reshape(-1).detach().cpu()
    valid = (target >= 0) & (target < num_classes)
    index = target[valid] * num_classes + pred[valid]
    bincount = torch.bincount(index, minlength=num_classes * num_classes)
    confusion += bincount.reshape(num_classes, num_classes).numpy()


def compute_metrics(confusion):
    tp = np.diag(confusion).astype(np.float64)
    gt = confusion.sum(axis=1).astype(np.float64)
    pred = confusion.sum(axis=0).astype(np.float64)
    iou = tp / np.maximum(gt + pred - tp, 1.0)
    oa = tp.sum() / max(confusion.sum(), 1.0)
    miou = float(np.mean(iou))
    return float(oa), miou, iou.tolist()


def compute_class_weights(train_set, num_classes, power=0.5, max_weight=5.0):
    counts = np.zeros(num_classes, dtype=np.float64)
    for path in train_set.files:
        labels = np.load(path, mmap_mode="r")[:, 6].astype(np.int64)
        counts += np.bincount(labels[(labels >= 0) & (labels < num_classes)], minlength=num_classes)
    freq = counts / max(counts.sum(), 1.0)
    weights = np.power(np.maximum(freq, 1e-12), -float(power))
    weights = weights / np.mean(weights)
    weights = np.minimum(weights, float(max_weight))
    weights = weights / np.mean(weights)
    return torch.from_numpy(weights.astype(np.float32)), counts


def ce_loss(logits, labels, args):
    return F.cross_entropy(logits, labels, weight=getattr(args, "class_weights", None))


def apply_eval_vote(points, vote_idx, vote_count):
    if vote_idx == 0:
        return points
    out = points.clone()
    angle = 2.0 * math.pi * float(vote_idx) / float(max(vote_count, 1))
    c, s = math.cos(angle), math.sin(angle)
    xyz = out[:, 0:3, :]
    x = xyz[:, 0, :].clone()
    y = xyz[:, 1, :].clone()
    xyz[:, 0, :] = c * x - s * y
    xyz[:, 1, :] = s * x + c * y
    # Rotate normalized XYZ around room-center only for the horizontal plane.
    if out.shape[1] >= 9:
        XYZ = out[:, 6:9, :]
        X = (XYZ[:, 0, :] - 0.5).clone()
        Y = (XYZ[:, 1, :] - 0.5).clone()
        XYZ[:, 0, :] = c * X - s * Y + 0.5
        XYZ[:, 1, :] = s * X + c * Y + 0.5
    return out


def load_text_prototypes(path, device, expected_classes=13):
    if not path:
        return None
    obj = torch.load(path, map_location="cpu")
    tensor = obj["prototypes"] if isinstance(obj, dict) else obj
    if tensor.shape[0] != expected_classes:
        raise ValueError("Expected %d text prototypes, got %d" % (expected_classes, tensor.shape[0]))
    return tensor.to(device).float()


def make_loaders(args):
    train_set = S3DISBlockDataset(
        args.data_root, "train", args.test_area, args.num_points, args.pc_attribs,
        augment=args.augment, args=args,
    )
    val_set = S3DISBlockDataset(
        args.data_root, "val", args.test_area, args.num_points, args.pc_attribs,
        augment=False, args=args,
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
    )
    return train_set, val_set, train_loader, val_loader


def run_epoch(model, loader, optimizer, device, args, train=True):
    model.train(train)
    meter = Meter()
    confusion = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_idx, (points, labels) in enumerate(loader):
            if train and args.limit_train_batches > 0 and batch_idx >= args.limit_train_batches:
                break
            if (not train) and args.limit_val_batches > 0 and batch_idx >= args.limit_val_batches:
                break
            points = points.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if train or args.val_vote <= 1:
                logits, z_vis, z_text = model(points)
            else:
                logits_acc = None
                z_vis = z_text = None
                for vote_idx in range(args.val_vote):
                    vote_points = apply_eval_vote(points, vote_idx, args.val_vote)
                    vote_logits, _, _ = model(vote_points)
                    logits_acc = vote_logits if logits_acc is None else logits_acc + vote_logits
                logits = logits_acc / float(args.val_vote)
            loss = ce_loss(logits, labels, args)
            if train and args.visual_aux_weight > 0:
                loss = loss + args.visual_aux_weight * ce_loss(z_vis, labels, args)
            if train and args.text_aux_weight > 0 and z_text is not None:
                loss = loss + args.text_aux_weight * ce_loss(z_text, labels, args)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            pred = logits.argmax(dim=1)
            meter.update(loss.item(), points.shape[0])
            update_confusion(confusion, pred, labels, args.num_classes)
    oa, miou, iou = compute_metrics(confusion)
    return meter.avg, oa, miou, iou


def apply_partial_freeze(model, prefixes):
    if not prefixes:
        return 0, 0
    utonia = getattr(model.segmentor.encoder, "utonia_model", None)
    if utonia is None:
        return 0, 0
    total = frozen = 0
    for name, param in utonia.named_parameters():
        total += param.numel()
        if any(name.startswith(prefix) for prefix in prefixes):
            param.requires_grad = False
            frozen += param.numel()
    return frozen, total


def build_optimizer(model, args):
    backbone_params, adapter_params, head_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("segmentor.encoder.utonia_model"):
            backbone_params.append(param)
        elif name.startswith("segmentor.encoder.point_mapper") or name.startswith("segmentor.encoder.edge_mappers"):
            adapter_params.append(param)
        else:
            head_params.append(param)
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.backbone_lr, "name": "utonia_backbone"})
    if adapter_params:
        groups.append({"params": adapter_params, "lr": args.adapter_lr, "name": "adapter"})
    if head_params:
        groups.append({"params": head_params, "lr": args.head_lr, "name": "heads"})
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay), groups


def load_compatible_state_dict(model, state):
    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key in current and tuple(value.shape) == tuple(current[key].shape):
            compatible[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return missing, unexpected, skipped


def save_checkpoint(path, model, optimizer, epoch, best_miou, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_miou": best_miou,
            "args": {k: v for k, v in vars(args).items() if k != "class_weights"},
            "classes": S3DIS_ORDER,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/root/autodl-tmp/Datasets/S3DIS/blocks_bs1_s1")
    parser.add_argument("--dpa_root", default="/root/autodl-tmp/workspace/ptv3_fs/DPA")
    parser.add_argument("--save_dir", default="/root/autodl-tmp/workspace/llm_pointseg/outputs/utonia_textproto")
    parser.add_argument("--test_area", default="Area_5")
    parser.add_argument("--num_classes", type=int, default=13)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--pc_attribs", default="xyzrgbXYZ")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--step_size", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", type=str2bool, default=True)
    parser.add_argument("--aug_scale", type=float, default=0.0)
    parser.add_argument("--aug_rot", type=str2bool, default=True)
    parser.add_argument("--aug_mirror_prob", type=float, default=0.0)
    parser.add_argument("--aug_jitter", type=str2bool, default=True)
    parser.add_argument("--text_prototypes", default="")
    parser.add_argument("--text_weight", type=float, default=0.0)
    parser.add_argument("--text_logit_scale", type=float, default=10.0)
    parser.add_argument("--learnable_text_gate", type=str2bool, default=False)
    parser.add_argument("--per_class_text_gate", type=str2bool, default=False)
    parser.add_argument("--text_gate_init", type=float, default=0.02)
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--visual_aux_weight", type=float, default=0.0)
    parser.add_argument("--text_aux_weight", type=float, default=0.0)
    parser.add_argument("--class_balanced_loss", type=str2bool, default=False)
    parser.add_argument("--class_weight_power", type=float, default=0.5)
    parser.add_argument("--class_weight_max", type=float, default=5.0)
    parser.add_argument("--val_vote", type=int, default=1)
    parser.add_argument("--backbone_lr", type=float, default=None)
    parser.add_argument("--adapter_lr", type=float, default=None)
    parser.add_argument("--head_lr", type=float, default=None)
    parser.add_argument("--freeze_utonia_prefixes", default="")
    parser.add_argument("--limit_train_batches", type=int, default=0)
    parser.add_argument("--limit_val_batches", type=int, default=0)
    parser.add_argument("--edgeconv_widths", default="[[64,64], [64,64], [64,64]]")
    parser.add_argument("--pc_in_dim", type=int, default=9)
    parser.add_argument("--utonia_repo_path", default="/root/autodl-tmp/workspace/ptv3_fs/COSeg/model")
    parser.add_argument("--utonia_checkpoint_path", default="/root/autodl-tmp/workspace/ptv3_fs/COSeg/checkpoints/utonia/utonia.pth")
    parser.add_argument("--utonia_backbone_dim", type=int, default=576)
    parser.add_argument("--utonia_feat_dim", type=int, default=256)
    parser.add_argument("--utonia_grid_size", type=float, default=0.01)
    parser.add_argument("--utonia_freeze_encoder", type=str2bool, default=True)
    parser.add_argument("--utonia_enable_flash", type=str2bool, default=False)
    args = parser.parse_args()

    args.edgeconv_widths = ast.literal_eval(args.edgeconv_widths)
    args.pc_in_dim = len(args.pc_attribs)
    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, val_set, train_loader, val_loader = make_loaders(args)
    if args.class_balanced_loss:
        class_weights, class_counts = compute_class_weights(train_set, args.num_classes, args.class_weight_power, args.class_weight_max)
        args.class_weights = class_weights.to(device)
    else:
        class_counts = None
        args.class_weights = None
    text_prototypes = load_text_prototypes(args.text_prototypes, device, args.num_classes)
    if args.text_weight > 0 and text_prototypes is None:
        raise ValueError("--text_weight > 0 requires --text_prototypes")

    model = TextProtoSeg(args, args.num_classes, text_prototypes).to(device)
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected, skipped = load_compatible_state_dict(model, state)
        print("loaded resume_checkpoint", args.resume_checkpoint)
        print("missing keys", missing)
        print("unexpected keys", unexpected)
        print("skipped incompatible keys", skipped)
    freeze_prefixes = [x.strip() for x in args.freeze_utonia_prefixes.split(",") if x.strip()]
    frozen_params, utonia_params = apply_partial_freeze(model, freeze_prefixes)
    args.backbone_lr = args.lr if args.backbone_lr is None else args.backbone_lr
    args.adapter_lr = args.lr if args.adapter_lr is None else args.adapter_lr
    args.head_lr = args.lr if args.head_lr is None else args.head_lr
    optimizer, optim_groups = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    writer = SummaryWriter(args.save_dir) if SummaryWriter is not None else None

    print("device", device)
    print("train_blocks", len(train_set), "val_blocks", len(val_set), "classes", S3DIS_ORDER)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("text_weight", args.text_weight, "text_prototypes", args.text_prototypes or None)
    print("learnable_text_gate", args.learnable_text_gate, "per_class_text_gate", args.per_class_text_gate, "text_gate_init", args.text_gate_init, "resume", args.resume_checkpoint or None)
    print("class_balanced_loss", args.class_balanced_loss, "class_weights", None if args.class_weights is None else [round(float(x), 4) for x in args.class_weights.detach().cpu().tolist()])
    print("val_vote", args.val_vote, "freeze_prefixes", freeze_prefixes, "frozen_utonia %.2fM/%.2fM" % (frozen_params / 1e6, utonia_params / 1e6))
    print("optimizer_groups", [(g.get("name"), g["lr"], sum(p.numel() for p in g["params"]) / 1e6) for g in optim_groups])
    print("params total %.2fM trainable %.2fM" % (total_params / 1e6, trainable_params / 1e6))

    best_miou = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_oa, train_miou, _ = run_epoch(model, train_loader, optimizer, device, args, train=True)
        val_loss, val_oa, val_miou, val_iou = run_epoch(model, val_loader, optimizer, device, args, train=False)
        scheduler.step()
        elapsed = time.time() - t0
        print(
            "epoch %03d/%03d train_loss %.4f train_oa %.4f train_miou %.4f val_loss %.4f val_oa %.4f val_miou %.4f time %.1fs"
            % (epoch, args.epochs, train_loss, train_oa, train_miou, val_loss, val_oa, val_miou, elapsed),
            flush=True,
        )
        print("val_iou", " ".join("%s:%.4f" % (name, iou) for name, iou in zip(S3DIS_ORDER, val_iou)), flush=True)
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("miou/train", train_miou, epoch)
            writer.add_scalar("miou/val", val_miou, epoch)
            writer.add_scalar("oa/train", train_oa, epoch)
            writer.add_scalar("oa/val", val_oa, epoch)
        # Best-only checkpointing: full Utonia fine-tune checkpoints are large.
        # Do not write last.pth every epoch; keep best.pth for recovery/export.
        if val_miou > best_miou:
            best_miou = val_miou
            save_checkpoint(os.path.join(args.save_dir, "best.pth"), model, optimizer, epoch, best_miou, args)
            print("saved best %.4f" % best_miou, flush=True)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
