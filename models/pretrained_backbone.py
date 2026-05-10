import importlib
import os
import sys

import torch
import torch.nn as nn


class PretrainedBackbone(nn.Module):
    """Adapter that exposes a pretrained 3D encoder as a DGCNN-like backbone interface."""

    def __init__(self, args):
        super(PretrainedBackbone, self).__init__()
        self.args = args
        self.grid_size = float(args.backbone_grid_size)
        self.checkpoint_path = args.backbone_checkpoint_path
        self.backbone_dim = int(args.backbone_dim)
        self.dgcnn_out_dim = int(getattr(args, "backbone_feat_dim", getattr(args, "dgcnn_mlp_widths", [256])[-1]))
        self.backbone_freeze_encoder = bool(args.backbone_freeze_encoder)

        self.encoder_model = self._build_backbone_encoder(
            checkpoint_path=self.checkpoint_path,
            repo_path=args.backbone_repo_path,
            enable_flash=bool(args.backbone_enable_flash),
        )

        if self.backbone_freeze_encoder:
            for p in self.encoder_model.parameters():
                p.requires_grad = False
            self.encoder_model.eval()

        self.point_mapper = nn.Sequential(
            nn.Conv1d(self.backbone_dim, self.dgcnn_out_dim, 1, bias=False),
            nn.BatchNorm1d(self.dgcnn_out_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.edge_mappers = nn.ModuleList()
        for widths in args.edgeconv_widths:
            out_dim = int(widths[-1])
            self.edge_mappers.append(
                nn.Sequential(
                    nn.Conv1d(self.dgcnn_out_dim, out_dim, 1, bias=False),
                    nn.BatchNorm1d(out_dim),
                    nn.LeakyReLU(negative_slope=0.2),
                )
            )

    def _build_backbone_encoder(self, checkpoint_path, repo_path, enable_flash):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "Pretrained checkpoint not found: %s. "
                "Please set --backbone_checkpoint_path to a valid file." % checkpoint_path
            )

        loader = self._import_backbone_loader(repo_path)
        model = loader(
            name=checkpoint_path,
            custom_config={"enable_flash": enable_flash},
        )
        return model

    def _import_backbone_loader(self, repo_path):
        module_name = getattr(self.args, "backbone_loader_module", "model")
        attr_name = getattr(self.args, "backbone_loader_attr", "load")

        candidate_paths = []
        if repo_path:
            candidate_paths.append(os.path.abspath(repo_path))
        current_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_paths.append(os.path.abspath(os.path.join(current_dir, "..", "external_backbone")))

        for path in candidate_paths:
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)
            try:
                module = importlib.import_module(module_name)
                return getattr(module, attr_name)
            except Exception:
                continue

        raise ImportError(
            "Failed to import the pretrained 3D encoder loader. Set --backbone_repo_path, "
            "--backbone_loader_module, and --backbone_loader_attr for your local backend."
        )

    def _forward_single(self, x_single):
        # x_single: (C, N) -> point tensors are (N, C)
        feat = x_single.transpose(0, 1).contiguous()
        coord = feat[:, :3].contiguous()
        num_points = feat.shape[0]

        point_dict = {
            "coord": coord.float(),
            "feat": feat.float(),
            "batch": torch.zeros(num_points, device=feat.device, dtype=torch.long),
            "grid_size": self.grid_size,
        }

        if self.backbone_freeze_encoder:
            self.encoder_model.eval()
            with torch.no_grad():
                point = self.encoder_model(point_dict)
        else:
            point = self.encoder_model(point_dict)

        # Recover to the original point resolution.
        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent

        feat_out = point.feat
        if feat_out.shape[0] != num_points:
            raise RuntimeError(
                "Pretrained encoder output point count mismatch: got %d, expected %d."
                % (feat_out.shape[0], num_points)
            )
        return feat_out.transpose(0, 1).unsqueeze(0)

    def forward(self, x):
        # x: (B, C, N)
        batch_feats = [self._forward_single(x[b]) for b in range(x.shape[0])]
        encoder_feat = torch.cat(batch_feats, dim=0)  # (B, C_u, N)

        if encoder_feat.shape[1] != self.backbone_dim:
            raise RuntimeError(
                "Unexpected pretrained encoder feature dim %d (configured: %d). "
                "Set --backbone_dim correctly."
                % (encoder_feat.shape[1], self.backbone_dim)
            )

        point_feat = self.point_mapper(encoder_feat)
        edgeconv_outputs = [edge_mapper(point_feat) for edge_mapper in self.edge_mappers]
        return edgeconv_outputs, point_feat, None
