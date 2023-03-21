from functools import partial
# from smac.env import MultiAgentEnv, StarCraft2Env
from .multiagentenv import MultiAgentEnv
from .multicar_env import MultiCarEnv
from .flagenv import FlagEnv
from .freeenv import FreeEnv
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
# REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["multicar"] = partial(env_fn, env=MultiCarEnv)
REGISTRY["flag"] = partial(env_fn, env=FlagEnv)
REGISTRY["free"] = partial(env_fn, env=FreeEnv)

# if sys.platform == "linux":
#     os.environ.setdefault("SC2PATH",
#                           os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
