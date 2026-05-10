import os
import sys

import torch
import torch.nn as nn


class UtoniaBackbone(nn.Module):
    """Adapter that exposes Utonia as a DGCNN-like backbone interface."""

    def __init__(self, args):
        super(UtoniaBackbone, self).__init__()
        self.grid_size = float(args.utonia_grid_size)
        self.utonia_ckpt_path = args.utonia_checkpoint_path
        self.utonia_backbone_dim = int(args.utonia_backbone_dim)
        self.dgcnn_out_dim = int(getattr(args, "utonia_feat_dim", getattr(args, "dgcnn_mlp_widths", [256])[-1]))
        self.utonia_freeze_encoder = bool(args.utonia_freeze_encoder)

        self.utonia_model = self._build_utonia_encoder(
            checkpoint_path=self.utonia_ckpt_path,
            repo_path=args.utonia_repo_path,
            enable_flash=bool(args.utonia_enable_flash),
        )

        if self.utonia_freeze_encoder:
            for p in self.utonia_model.parameters():
                p.requires_grad = False
            self.utonia_model.eval()

        self.point_mapper = nn.Sequential(
            nn.Conv1d(self.utonia_backbone_dim, self.dgcnn_out_dim, 1, bias=False),
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

    def _build_utonia_encoder(self, checkpoint_path, repo_path, enable_flash):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "Utonia checkpoint not found: %s. "
                "Please set --utonia_checkpoint_path to a valid file." % checkpoint_path
            )

        loader = self._import_utonia_loader(repo_path)
        model = loader(
            name=checkpoint_path,
            custom_config={"enable_flash": enable_flash},
        )
        return model

    def _import_utonia_loader(self, repo_path):
        try:
            from utonia.model import load as utonia_load

            return utonia_load
        except Exception:
            pass

        candidate_paths = []
        if repo_path:
            candidate_paths.append(os.path.abspath(repo_path))
        current_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_paths.append(os.path.abspath(os.path.join(current_dir, "..", "..", "Utonia")))
        candidate_paths.append(os.path.abspath(os.path.join(current_dir, "..", "Utonia")))

        for path in candidate_paths:
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)
                try:
                    from utonia.model import load as utonia_load

                    return utonia_load
                except Exception:
                    continue

        raise ImportError(
            "Failed to import Utonia. Install the package or set --utonia_repo_path "
            "to the Utonia repository root."
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

        if self.utonia_freeze_encoder:
            self.utonia_model.eval()
            with torch.no_grad():
                point = self.utonia_model(point_dict)
        else:
            point = self.utonia_model(point_dict)

        # Recover to the original point resolution.
        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent

        feat_out = point.feat
        if feat_out.shape[0] != num_points:
            raise RuntimeError(
                "Utonia output point count mismatch: got %d, expected %d."
                % (feat_out.shape[0], num_points)
            )
        return feat_out.transpose(0, 1).unsqueeze(0)

    def forward(self, x):
        # x: (B, C, N)
        batch_feats = [self._forward_single(x[b]) for b in range(x.shape[0])]
        utonia_feat = torch.cat(batch_feats, dim=0)  # (B, C_u, N)

        if utonia_feat.shape[1] != self.utonia_backbone_dim:
            raise RuntimeError(
                "Unexpected Utonia feature dim %d (configured: %d). "
                "Set --utonia_backbone_dim correctly."
                % (utonia_feat.shape[1], self.utonia_backbone_dim)
            )

        point_feat = self.point_mapper(utonia_feat)
        edgeconv_outputs = [edge_mapper(point_feat) for edge_mapper in self.edge_mappers]
        return edgeconv_outputs, point_feat, None
