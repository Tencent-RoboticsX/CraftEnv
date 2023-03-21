import gym
from gym.spaces import Tuple as GymTuple

from craftenv.sim_envs.pybullet_envs.craft.craft_env import CraftEnv


class SingleAgentWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SingleAgentWrapper, self).__init__(env)

        self.observation_space = GymTuple([env.observation_space])
        self.action_space = GymTuple([env.action_space])

    def reset(self, **kwargs):
        obs = self.env.reset()
        return (obs,)

    def step(self, action):
        obs, rwd, done, info = super(SingleAgentWrapper, self).step(action[0])
        if "post_process_data" in info:
            info["post_process_data"] = (info["post_process_data"],)
        return (obs,), (rwd,), done, info


def create_pybullet_env(**env_config):
    arena_id = env_config["arena_id"]
    assert arena_id in [
        "Craft-v0"
    ]
    enable_render = env_config["render"] if "render" in env_config else False

    enable_render = env_config["render"] if "render" in env_config else False

    def create_single_env():
        if arena_id in ["Craft-v0"]:
            init_blueprint_path = list(
                env_config["init_blueprint_path"].split(","))
            env0 = CraftEnv(enable_render, init_blueprint_path, env_config)
        else:
            raise NotImplementedError
        env0 = SingleAgentWrapper(env0)
        return env0
    env = create_single_env()
    return env
