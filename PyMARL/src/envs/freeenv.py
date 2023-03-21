from .multiagentenv import MultiAgentEnv
from craft.free_env import FreeEnv as Env


class FreeEnv(MultiAgentEnv):

    def __init__(self, **env_config):
        enable_render = env_config["render"] if "render" in env_config else False
        init_blueprint_path = list(env_config['init_blueprint_path'].split(','))
        self.env = Env(enable_render, init_blueprint_path, env_config)
        self.env.reset()
        self.episode_limit = env_config["max_steps"] + 10
        self.n_agents = self.env._blackboard.smartcar_num
        return

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        return reward, done, info

    def get_obs(self):
        obs_tuple = self.env.get_obs()
        obs_list = list(obs_tuple)
        result = [obs_list[i][0] for i in range(0, len(obs_list))]
        return result

    def get_obs_agent(self, agent_id):
        all_obs = self.get_obs()
        return all_obs[agent_id]

    def get_obs_size(self):
        return self.env.ob_dim

    def get_state(self):
        obs_tuple = self.get_obs()
        obs_list = list(obs_tuple)
        global_state = obs_list[0][self.env._ob_dim:]
        return global_state

    def get_state_size(self):
        state = self.get_state()
        return len(state)

    def get_avail_actions(self):
        obs_tuple = self.env.get_obs()
        obs_list = list(obs_tuple)
        result = [obs_list[i][1] for i in range(0, len(obs_list))]
        return result

    def get_avail_agent_actions(self, agent_id):
        all_masks = self.get_avail_actions()
        result = all_masks[agent_id]
        return result

    def get_total_actions(self):
        return self.env.ac_dim

    def reset(self):
        return self.env.reset()

    def render(self):
        return None

    def close(self):
        return

    def seed(self):
        return 0

    def save_replay(self):
        return

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        return {}
