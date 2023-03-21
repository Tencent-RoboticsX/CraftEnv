import itertools
import math

import numpy as np
import yaml

from .craft_env import CraftEnv
from .grid_objs import ObjType


class GoalEnv(CraftEnv):

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
        self.goal_obs = self.generate_goal_obs(env_config["design_path"])
        self.design_length = 0
        for k in self.design_dict.keys():
            self.design_length += len(self.design_dict[k])
        self.last_pos_dict = None
        print("what the hell")

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

    def generate_goal_obs(self, design_path):
        design_dict = None
        with open(design_path) as f:
            source = yaml.load(f, Loader=yaml.loader.SafeLoader)
            design_dict = {
                "block": [],
                "unfolded_body": [],
                "unfolded_foot": []
            }
            for key in self.key_mapping.keys():
                if key not in source.keys():
                    continue
                for obj in source[key]:
                    x, y, z = int(obj['x']), int(obj['y']), int(obj['z'])
                    design_dict[key].append((x, y, z))
        # obs structure:
        # [x, y, z, smartcar, foldslope, unfoldbody, unfoldfoot,
        # block, flag, goal, dir0, dir1, is_lift]

        def decide_yaw(info_dict, x, y, z):
            yaw = -1
            assert (x, y, z) in info_dict["unfolded_body"]
            if (x + 1, y, z) in info_dict["unfolded_foot"]:
                yaw = 0
            if (x, y + 1, z) in info_dict["unfolded_foot"]:
                yaw = 1
            if (x - 1, y, z) in info_dict["unfolded_foot"]:
                yaw = 2
            if (x, y - 1, z) in info_dict["unfolded_foot"]:
                yaw = 3
            assert yaw != -1
            return yaw

        def convert_yaw(yaw):
            converter = [(0, 0), (0, 1), (1, 0), (1, 1)]
            return converter[yaw][0], converter[yaw][1]

        # Note that the obs should be stacked in the correct sequence!
        # 1. unfolded_body
        # 2. unfolded_foot
        # 3. block
        obs_unfolded_body = []
        obs_unfolded_foot = []
        obs_block = []
        for (x, y, z) in design_dict["unfolded_body"]:
            yaw = decide_yaw(design_dict, x, y, z)
            dir0, dir1 = convert_yaw(yaw)
            obs_unfolded_body.append(
                [x, y, z, 0, 0, 1, 0, 0, 0, 0, dir0, dir1, 0])
            # locate the foot
            xf, yf, zf = x, y, z
            if yaw == 0:
                xf += 1
            if yaw == 1:
                yf += 1
            if yaw == 2:
                xf -= 1
            if yaw == 3:
                yf -= 1
            obs_unfolded_foot.append(
                [xf, yf, zf, 0, 0, 1, 0, 0, 0, 0, dir0, dir1, 0])
        for (x, y, z) in design_dict["block"]:
            obs_block.append([x, y, z, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        obs_goal = np.array(obs_unfolded_body + obs_unfolded_foot + obs_block,
                            dtype='float')
        return obs_goal

    def reset(self):
        self.design_list, self.design_dict = self.read_design(
            self.env_config["design_path"])
        self.last_pos_dict = None
        obs = super().reset()
        obs = self.obs_postprocess(obs)
        return obs

    def _calculate_diff(self, pos_dict):
        design_dict = self.design_dict
        hit = 0
        diff = 0
        for key in design_dict:
            target_pos_list = design_dict[key]
            for target_pos in target_pos_list:
                if target_pos in pos_dict[key]:
                    hit += 1
                    diff += 1
                # add for two-layer reward
                if key == "block" and target_pos[2] == 2:
                    diff += 1
        # add for complete the goal
        if hit == self.design_length:
            diff += 8
        return diff

    def _calculate_dist(self, obj_pos, obj_typename):
        assert obj_typename in ["block"]
        reward = 10000
        design_dict = self.design_dict
        for pos in design_dict[obj_typename]:
            dist = math.sqrt((pos[0] - obj_pos[0])**2 +
                             (pos[1] - obj_pos[1])**2 +
                             (pos[2] - obj_pos[2])**2)
            if dist < reward:
                reward = dist
        return reward

    def _compute_reward(self, blackboard=None):
        reward = 0
        before = 0
        after = 0
        if self.last_pos_dict is not None:
            before = self._calculate_diff(self.last_pos_dict)
            reward -= before
        current_pos_dict = self.get_pos_dict()
        after = self._calculate_diff(current_pos_dict)
        reward += after
        self.last_pos_dict = current_pos_dict
        return reward

    def obs_postprocess(self, obs):
        goal_obs = self.goal_obs
        fake_action_mask = [False for _ in range(15)]
        obs += (goal_obs, fake_action_mask)
        return obs

    def step(self, action):
        enable_local_obs = self.env_config.get('enable_local_obs', False)
        if not enable_local_obs:
            obs, reward, done, info = super().step(action)
            obs = self.obs_postprocess(obs)
            return obs, reward, done, info
        else:
            return NotImplementedError
