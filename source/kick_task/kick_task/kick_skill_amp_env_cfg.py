from isaaclab.utils import configclass

from beyondAMP.obs_groups import AMPObsBaiscCfg

from .kick_skill_env_cfg import KickSkillEnvCfg
from .kick_skill_env_cfg import KickSkillPlayEnvCfg
from .kick_skill_env_cfg import ObservationsCfg
from .kick_skill_bootstrap_env_cfg import KickSkillBootstrapEnvCfg
from .kick_skill_bootstrap_env_cfg import KickSkillBootstrapPlayEnvCfg
from .kick_skill_bootstrap_env_cfg import BootstrapObservationsCfg


@configclass
class KickSkillAmpObservationsCfg(ObservationsCfg):
    amp: AMPObsBaiscCfg = AMPObsBaiscCfg()


@configclass
class KickSkillAmpEnvCfg(KickSkillEnvCfg):
    observations: KickSkillAmpObservationsCfg = KickSkillAmpObservationsCfg()


@configclass
class KickSkillAmpPlayEnvCfg(KickSkillPlayEnvCfg):
    observations: KickSkillAmpObservationsCfg = KickSkillAmpObservationsCfg()


@configclass
class KickSkillBootstrapAmpObservationsCfg(BootstrapObservationsCfg):
    amp: AMPObsBaiscCfg = AMPObsBaiscCfg()


@configclass
class KickSkillBootstrapAmpEnvCfg(KickSkillBootstrapEnvCfg):
    observations: KickSkillBootstrapAmpObservationsCfg = KickSkillBootstrapAmpObservationsCfg()


@configclass
class KickSkillBootstrapAmpPlayEnvCfg(KickSkillBootstrapPlayEnvCfg):
    observations: KickSkillBootstrapAmpObservationsCfg = KickSkillBootstrapAmpObservationsCfg()
