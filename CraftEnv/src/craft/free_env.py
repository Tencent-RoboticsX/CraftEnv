import itertools

import numpy as np
import yaml

from .craft_env import CraftEnv
from .grid_objs import ObjType


class FreeEnv(CraftEnv):

    def __init__(self, enable_render, init_blueprint_path, env_config):
        super().__init__(enable_render, init_blueprint_path, env_config)
        self.key_mapping = {
            "block": 1,
            "folded_slope": 2,
            "unfolded_body": 3,
            "unfolded_foot": 4
        }
        self.design_list, self.design_dict = self.read_design(
            env_config["design_path"])
        ##########
        temp = 0
        for k in self.design_dict.keys():
            temp += len(self.design_dict[k])
        self.design_length = temp
        print("Direct load from pymarl succeed")
        print("design_length", self.design_length)
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
        self.design_list, self.design_dict = self.read_design(
            self.env_config["design_path"])
        self.last_pos_dict = None
        obs = super().reset()
        return obs

    def compute_score(self, pos_dict):
        blocks = pos_dict["block"]
        visited = {block: False for block in blocks}
        counter = [0 for _ in range(0, len(blocks))]
        index = 0
        for block in blocks:
            if visited[block] is True:
                continue
            else:
                visited[block] = True
                counter[index] = 1
                stack = [block]
                while len(stack) != 0:
                    top = stack.pop()
                    neighbors = [(top[0] - 1, top[1], top[2]),
                                 (top[0] + 1, top[1], top[2]),
                                 (top[0], top[1] - 1, top[2]),
                                 (top[0], top[1] + 1, top[2])]
                    for neighbor in neighbors:
                        if neighbor in blocks and visited[neighbor] is False:
                            visited[neighbor] = True
                            counter[index] += 1
                            stack.append(neighbor)
                index += 1
        score = 0
        for c in counter:
            if c == 0:
                break
            if c == 1:
                continue
            elif c > score:
                score = c
        return score

    def _compute_reward(self, blackboard=None):
        reward = 0
        before = 0
        after = 0
        if self.last_pos_dict is not None:
            before = self.compute_score(self.last_pos_dict)
            reward -= before
        current_pos_dict = self.get_pos_dict()
        after = self.compute_score(current_pos_dict)
        reward += after
        self.last_pos_dict = current_pos_dict
        return reward

    def step(self, action):
        enable_local_obs = self.env_config.get('enable_local_obs', False)
        if not enable_local_obs:
            obs, reward, done, info = super().step(action)
            return obs, reward, done, info
        else:
            return NotImplementedError
