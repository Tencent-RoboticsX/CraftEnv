import copy
import random
import time

import gym
import numpy as np
import pybullet
from pybullet_utils import bullet_client

from craft import utils

from .action_mask import ActionMask
from .blackboard import Blackboard
from .goal import Goal
from .matrix_to_bullet import MatrixToBullet
from .planner import BreadthFirstSearch
from .scene import Scene
from .smartcar import Smartcar


class MatrixEnv(gym.Env):

    def __init__(self, enable_render, init_blueprint_path, env_config):
        self.enable_render = enable_render
        self._blackboard = Blackboard(init_blueprint_path)
        search_depth = env_config.get('search_depth', 10)
        self.smartcars = [
            Smartcar(self._blackboard, i, search_depth=search_depth)
            for i in range(self._blackboard.smartcar_num)
        ]
        self.scene = Scene(self._blackboard)

        self._blackboard.smartcars = self.smartcars
        self._blackboard.scene = self.scene
        self._blackboard.action_mask_proxy = ActionMask(self._blackboard)
        self._blackboard.bfs = BreadthFirstSearch(self._blackboard)

        self._blackboard.goal = Goal(self._blackboard)

        if self.enable_render:
            self._bullet_client = bullet_client.BulletClient(
                connection_mode=pybullet.GUI)
        else:
            self._bullet_client = bullet_client.BulletClient(
                connection_mode=pybullet.DIRECT)
        self._blackboard._bullet_client = self._bullet_client
        self.matrix_to_bullet = MatrixToBullet(self._bullet_client,
                                               self._blackboard)
        self._step = 0
        self.env_config = env_config
        if "work_mode" in self.env_config:
            # 0: train mode, 1: record mode, 2: play mode
            self.work_mode = self.env_config["work_mode"]
        else:
            self.work_mode = 0
        self._total_step = 0

    def predict(self, action: list, blackboard=None):
        """predict the observation after an action with specific blackboard."""
        predict_blackboard = blackboard
        if predict_blackboard is None:
            predict_blackboard = copy.deepcopy(self._blackboard)

        reward = 0
        done = False
        action_order = np.arange(predict_blackboard.smartcar_num)
        random.shuffle(action_order)
        for i in action_order:
            predict_blackboard.smartcars[i].step(action[i])
            if done:
                reward = 1
                break
        done = self._is_done(predict_blackboard)
        obs = self.get_obs(predict_blackboard)
        info = {predict_blackboard}
        return obs, reward, done, info

    def reset(self):
        blueprint = random.choice(self._blackboard.blueprint_path_list)
        self._blackboard.reset(blueprint)
        self._blackboard.goal.reset()
        self.scene.reset()
        for smartcar in self.smartcars:
            smartcar.reset()
        self.matrix_to_bullet.sync()
        if self.work_mode == 1:
            self.record_action = []
            self.record_action_order = []
        elif self.work_mode == 2:
            assert "yaml_save_path" in self.env_config, \
                "yaml_save_path arg is needed in play mode"
            assert "action_yaml_path" in self.env_config, \
                "action_yaml_path arg is needed in play mode"
            self.record_action, self.record_action_order = \
                utils.load_action2yaml(self._blackboard.smartcar_num,
                                       self.env_config['action_yaml_path'])
        self._step = 0

        return self.get_obs()

    def step(self, action: list):
        done = False
        action_order = np.arange(self._blackboard.smartcar_num)
        random.shuffle(action_order)

        if self.work_mode == 2:
            try:
                action, action_order = self.record_action[
                    self._step], self.record_action_order[self._step]
            except IndexError:
                print("record actions reach limit.")
                return None, 0, False, {}

        for i in action_order:
            self.smartcars[i].step(action[i])

        obs = self.get_obs()
        reward = self._compute_reward()
        done = self._is_done()
        info = {}

        self.matrix_to_bullet.sync()
        if self.work_mode == 1:
            assert "yaml_save_path" in self.env_config, \
                "yaml_save_path arg is needed in record mode"
            timestamp = time.strftime("%b-%d-%H:%M:%S", time.localtime())
            utils.save_scene2yaml(self._blackboard, self._step, timestamp,
                                  self.env_config["yaml_save_path"])
            self.record_action.append(action)
            self.record_action_order.append(action_order)
            if self._step == 0:
                self.act_timestamp = timestamp
            utils.save_action2yaml(self.record_action,
                                   self.record_action_order,
                                   self.act_timestamp,
                                   self.env_config["yaml_save_path"])

        self._step += 1
        self._total_step += 1
        return obs, reward, done, info

    def get_obs(self, blackboard=None):
        raise NotImplementedError

    def _is_done(self, blackboard=None):
        """Indicates whether or not the episode is done."""
        return False

    def _compute_reward(self, blackboard=None):
        """Calculates the reward to give based on the observations given."""
        return 1

    def get_flag_pos(self, blackboard=None):
        raise NotImplementedError
