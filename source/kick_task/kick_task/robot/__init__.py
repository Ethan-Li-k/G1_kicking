"""Robot configurations inlined from robotlib (https://github.com/Renforce-Dynamics/robotlib)."""

import os

ASSET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "assets"))

from .g1_cfg import G1_CYLINDER_CFG, G1_OPENSOURCE_CFG
from .g1_keys import g1_key_body_names, g1_anchor_name
