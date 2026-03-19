"""Data references for kick-skill AMP training (kick-only dataset)."""

import os
import numpy as np

from kick_task.robot import g1_anchor_name as _g1_anchor_name
from kick_task.robot import g1_key_body_names as _g1_key_body_names


_DEFAULT_KICK_LIST = "data/datasets/g1_kick_skill/motion_files_kick.txt"
_DEFAULT_KICK_DIR = "data/datasets/g1_kick_skill"
_FALLBACK_SHOOT_LIST = "data/datasets/g1_shoot_stage2/motion_files_stage2.txt"

_KICK_LIST = os.getenv("AMP_KICK_MOTION_LIST", _DEFAULT_KICK_LIST).strip()
_KICK_DIR = os.getenv("AMP_KICK_MOTION_DIR", _DEFAULT_KICK_DIR).strip()
_UNIFORM_MOTION_WEIGHT = float(os.getenv("AMP_KICK_UNIFORM_MOTION_WEIGHT", "1.0"))
_ANTI_TOE_FILTER = os.getenv("AMP_ANTI_TOE_FILTER", "1") == "1"
_ANTI_TOE_DROP_CLIP = os.getenv("AMP_ANTI_TOE_DROP_CLIP", "0") == "1"
_ANTI_TOE_BAD_RATIO_TH = float(os.getenv("AMP_ANTI_TOE_BAD_RATIO_TH", "0.15"))
_ANTI_TOE_DOWNWEIGHT = float(os.getenv("AMP_ANTI_TOE_DOWNWEIGHT", "0.35"))

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_CANDIDATE_PREFIXES = [
    "",
    _REPO_ROOT,
]
_REQUIRED_MOTION_KEYS = {
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
}


def _resolve_existing_path(path: str) -> str:
    for prefix in _CANDIDATE_PREFIXES:
        candidate = os.path.join(prefix, path) if prefix else path
        if os.path.isfile(candidate):
            return candidate
    return path


def _resolve_existing_dir(path: str) -> str:
    for prefix in _CANDIDATE_PREFIXES:
        candidate = os.path.join(prefix, path) if prefix else path
        if os.path.isdir(candidate):
            return candidate
    return path


def _is_valid_motion_file(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        with np.load(path) as data:
            return _REQUIRED_MOTION_KEYS.issubset(set(data.files))
    except Exception:
        return False


def _load_motion_files(list_path: str) -> list[str]:
    resolved_list_path = _resolve_existing_path(list_path)
    if not os.path.isfile(resolved_list_path):
        return []

    motion_files: list[str] = []
    list_dir = os.path.dirname(resolved_list_path)
    with open(resolved_list_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            rel = line.strip()
            if rel and not rel.startswith("#"):
                if os.path.isabs(rel):
                    if _is_valid_motion_file(rel):
                        motion_files.append(rel)
                    continue

                candidates = [
                    _resolve_existing_path(rel),
                    os.path.join(list_dir, rel),
                    os.path.join(os.path.dirname(list_dir), rel),
                ]
                resolved = next((path for path in candidates if os.path.isfile(path)), candidates[0])
                if _is_valid_motion_file(resolved):
                    motion_files.append(resolved)
    return motion_files


def _load_motion_files_from_dir(dir_path: str) -> list[str]:
    resolved_dir_path = _resolve_existing_dir(dir_path)
    if not os.path.isdir(resolved_dir_path):
        return []

    motion_files: list[str] = []
    for name in sorted(os.listdir(resolved_dir_path)):
        if not name.endswith(".npz"):
            continue
        full_path = os.path.join(resolved_dir_path, name)
        if _is_valid_motion_file(full_path):
            motion_files.append(full_path)
    return motion_files


def _build_uniform_motion_weights(motion_files: list[str], default_weight: float = 1.0) -> dict[str, float]:
    return {
        os.path.splitext(os.path.basename(path))[0]: float(default_weight)
        for path in motion_files
    }


def _quat_apply_np_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Apply quaternion rotation to vectors. quat=[w,x,y,z], vec=[...,3]."""
    q_xyz = quat[..., 1:4]
    qw = quat[..., 0:1]
    t = 2.0 * np.cross(q_xyz, vec)
    return vec + qw * t + np.cross(q_xyz, t)


def _estimate_bad_toe_frame_ratio(npz_path: str) -> float:
    """Estimate bad-frame ratio for right-foot toe-bias in a clip.

    Criteria (if available):
    - right_ankle_pitch > 12deg
    - heel_h - toe_h > 0.015
    - foot_parallel_cos < 0.92
    """
    try:
        with np.load(npz_path) as data:
            bad_flags: list[np.ndarray] = []

            # 1) Right ankle pitch threshold.
            if "joint_pos" in data and "joint_names" in data:
                joint_names = [str(x) for x in data["joint_names"].tolist()]
                if "right_ankle_pitch_joint" in joint_names:
                    j = joint_names.index("right_ankle_pitch_joint")
                    q = data["joint_pos"][:, j]
                    bad_flags.append(np.abs(q) > np.deg2rad(12.0))

            # 2) Heel-toe height and foot parallel from right ankle roll link pose.
            if "body_names" in data and "body_pos_w" in data and "body_quat_w" in data:
                body_names = [str(x) for x in data["body_names"].tolist()]
                if "right_ankle_roll_link" in body_names:
                    b = body_names.index("right_ankle_roll_link")
                    pos = data["body_pos_w"][:, b, :]  # [T,3]
                    quat = data["body_quat_w"][:, b, :]  # [T,4], assumed [w,x,y,z]

                    toe_off = np.zeros_like(pos)
                    heel_off = np.zeros_like(pos)
                    toe_off[:, 0] = 0.11
                    heel_off[:, 0] = -0.09
                    toe_h = (pos + _quat_apply_np_wxyz(quat, toe_off))[:, 2]
                    heel_h = (pos + _quat_apply_np_wxyz(quat, heel_off))[:, 2]
                    bad_flags.append((heel_h - toe_h) > 0.015)

                    z_axis = np.zeros_like(pos)
                    z_axis[:, 2] = 1.0
                    z_w = _quat_apply_np_wxyz(quat, z_axis)
                    bad_flags.append(z_w[:, 2] < 0.92)

            if len(bad_flags) == 0:
                return 0.0

            bad = bad_flags[0].copy()
            for f in bad_flags[1:]:
                bad = np.logical_or(bad, f)
            return float(np.mean(bad))
    except Exception:
        return 0.0


kick_motion_files = _load_motion_files(_KICK_LIST)
if not kick_motion_files:
    kick_motion_files = _load_motion_files_from_dir(_KICK_DIR)

if not kick_motion_files:
    kick_motion_files = _load_motion_files(_FALLBACK_SHOOT_LIST)

if not kick_motion_files:
    kick_motion_files = [_resolve_existing_path("data/datasets/g1_kick_skill/placeholder.npz")]

kick_motion_data_weights = _build_uniform_motion_weights(kick_motion_files, _UNIFORM_MOTION_WEIGHT)

if _ANTI_TOE_FILTER and kick_motion_data_weights:
    filtered_weights: dict[str, float] = {}
    dropped = 0
    downweighted = 0
    for path in kick_motion_files:
        key = os.path.splitext(os.path.basename(path))[0]
        w = float(kick_motion_data_weights.get(key, _UNIFORM_MOTION_WEIGHT))
        ratio = _estimate_bad_toe_frame_ratio(path)
        if ratio > _ANTI_TOE_BAD_RATIO_TH:
            if _ANTI_TOE_DROP_CLIP:
                dropped += 1
                continue
            w *= _ANTI_TOE_DOWNWEIGHT
            downweighted += 1
        filtered_weights[key] = w

    # Avoid empty dict if everything got dropped by a strict filter.
    if filtered_weights:
        kick_motion_data_weights = filtered_weights
    print(
        f"[kick_amp_data_cfg] anti-toe filter enabled: "
        f"clips={len(kick_motion_files)}, dropped={dropped}, downweighted={downweighted}"
    )

g1_key_body_names = _g1_key_body_names
g1_anchor_name = _g1_anchor_name
