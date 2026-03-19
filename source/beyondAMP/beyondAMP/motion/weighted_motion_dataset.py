from __future__ import annotations

import os
import re
import torch
from typing import List, Literal
from isaaclab.utils import configclass
from .motion_dataset import MotionDataset, MotionDatasetCfg

class WeightedMotionDataset(MotionDataset):
    """
    Extend MotionDataset with weighted sampling on transition pairs (t, t+1).
    """

    def __init__(
        self,
        cfg: MotionDatasetCfg,
        env,
        device="cpu",
        traj_weights: List[float] | None = None,
        transition_weights: torch.Tensor | None = None,
    ):
        super().__init__(cfg, env, device)
        num_transitions = len(self.index_t)
        if traj_weights is None:
            traj_weights = self._build_traj_weights_from_cfg(cfg)

        if transition_weights is None and traj_weights is not None:
            transition_weights = self._build_transition_weights_from_traj(traj_weights)

        if transition_weights is not None:
            assert transition_weights.shape[0] == num_transitions
            self.weights = transition_weights.to(device).clone()
        else:
            self.weights = torch.ones(len(self.index_t)).to(device)

        self._traj_weights = traj_weights
        self.norm_weights()

    @staticmethod
    def _normalize_motion_key(name: str) -> str:
        key = os.path.splitext(os.path.basename(name))[0]
        key = key.strip().replace(" ", "_")
        key = re.sub(r"_+", "_", key)
        return key

    def _build_traj_weights_from_cfg(self, cfg: MotionDatasetCfg) -> List[float] | None:
        motion_data_weights = getattr(cfg, "motion_data_weights", None)
        if not motion_data_weights:
            return None

        normalized_weight_map = {
            self._normalize_motion_key(k): float(v) for k, v in motion_data_weights.items()
        }

        traj_weights: List[float] = []
        for motion_file in self.motion_files:
            basename = os.path.basename(motion_file)
            stem = os.path.splitext(basename)[0]

            if stem in motion_data_weights:
                traj_weights.append(float(motion_data_weights[stem]))
                continue
            if basename in motion_data_weights:
                traj_weights.append(float(motion_data_weights[basename]))
                continue

            normalized_stem = self._normalize_motion_key(stem)
            traj_weights.append(float(normalized_weight_map.get(normalized_stem, 1.0)))

        return traj_weights

    # ---------------------------------------------------------
    # Normalization
    # ---------------------------------------------------------
    def norm_weights(self):
        self.weights = self.weights / (self.weights.sum() + 1e-9)

    def update_weights(self, weights: torch.Tensor, method: Literal["sum", "mean", "replace"]="sum", inplace=True):
        if method in ["sum", "mean"]:
            self.weights += weights
            self.norm_weights()
        elif method == "replace":
            self.weights.copy_(weights.to(self.device))
            self.norm_weights()
        else:
            raise NotImplementedError(f"Method: {method} not implemented.")

    # ------------------------------------------------------------------
    # Build from trajectory-level weights
    # ------------------------------------------------------------------
    def _build_transition_weights_from_traj(self, traj_weights):
        """
        Convert trajectory-level weights to transition-level weights.

        Params:
            traj_weights (List[float] or None)

        Returns:
            Tensor of shape (#transitions,)
        """
        if traj_weights is None:
            return torch.ones(len(self.index_t))

        traj_weights = torch.tensor(traj_weights, dtype=torch.float32)

        weights = []
        for w, L in zip(traj_weights, self.traj_lengths):
            if L >= 2:
                # L frames → L-1 transitions
                weights.append(torch.full((L - 1,), float(w)))

        return torch.cat(weights, dim=0)

    # ---------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------
    def sample_batch(self, batch_size: int, replacement = True):
        idx = torch.multinomial(self.weights, batch_size, replacement=replacement)
        t   = self.index_t[idx]
        tp1 = self.index_tp1[idx]
        return t, tp1


@configclass
class WeightedMotionDatasetCfg(MotionDatasetCfg):
    class_type: type[WeightedMotionDataset] = WeightedMotionDataset
