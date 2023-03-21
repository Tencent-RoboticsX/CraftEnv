import itertools

import numpy as np
import yaml

from .flag_craft_env import CraftEnv
from .grid_objs import ObjType


class FlagEnv(CraftEnv):

    def __init__(self, enable_render, init_blueprint_path, env_config):
        super().__init__(enable_render, init_blueprint_path, env_config)
        self.key_mapping = {
            "block": 1,
            "folded_slope": 2,
            "unfolded_body": 3,
            "unfolded_foot": 4
        }
        ##########
        self.hit = False
        self.current_step = -1
        self.last_distance = None
        ##########
        self.last_pos_dict = None

    def read_design(self, design_path):
        with open(design_path) as f:
            source = yaml.load(f, Loader=yaml.loader.SafeLoader)
            design_list = np.zeros(self.area_size)
            design_dict = {
                "block": [],
                "folded_slope": [],
                "unfolded_body": [],
                "unfolded_foot": []
            }
            for key in self.key_mapping.keys():
                if key not in source.keys():
                    continue
                for obj in source[key]:
                    x, y, z = int(obj['x']), int(obj['y']), int(obj['z'])
                    design_list[x][y][z] = self.key_mapping[key]
                    design_dict[key].append((x, y, z))
        return design_list, design_dict

    def get_pos_list(self):
        raise NotImplementedError

    def get_pos_dict(self):
        result = {
            "block": [],
            "folded_slope": [],
            "unfolded_body": [],
            "unfolded_foot": []
        }
        grid = self._blackboard.grid
        for i, j, k in itertools.product(range(self.area_size[0]),
                                         range(self.area_size[1]),
                                         range(self.area_size[2])):
            obj = grid[i][j][k]
            if obj.type is ObjType.Block or obj.type is \
                    ObjType.FoldedSlopeGear:
                result["block"].append((i, j, k))
            elif obj.type is ObjType.FoldedSlope:
                result["folded_slope"].append((i, j, k))
            elif obj.type is ObjType.UnfoldedSlopeBody:
                result["unfolded_body"].append((i, j, k))
            elif obj.type is ObjType.UnfoldedSlopeFoot:
                result["unfolded_foot"].append((i, j, k))
            else:
                pass
        return result

    def reset(self):
        self.hit = False
        self.current_step = -1
        self.last_distance = None
        obs = super().reset()
        return obs

    def _calculate_dist(self):
        flag = (self.flag_pos_x, self.flag_pos_y, self.flag_pos_z)
        goal = (self._blackboard.goal.x, self._blackboard.goal.y,
                self._blackboard.goal.z)
        dist = abs(flag[0] - goal[0]) + abs(flag[1] - goal[1]) + abs(flag[2] -
                                                                     goal[2])
        return dist

    def _compute_reward(self, blackboard=None):
        reward = None
        self.current_step += 1
        if self.last_distance is None:
            dist = self._calculate_dist()
            reward = 0
            self.last_distance = dist
        else:
            dist = self._calculate_dist()
            reward = self.last_distance - dist
            self.last_distance = dist
            if dist == 0 and (self.hit is False):
                self.hit = True
                reward += (20 - self.current_step)
        return reward

    def step(self, action):
        enable_local_obs = self.env_config.get('enable_local_obs', False)
        if not enable_local_obs:
            obs, reward, done, info = super().step(action)
            return obs, reward, done, info
        else:
            return NotImplementedError
