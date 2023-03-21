import itertools
import time

import numpy as np
from gym import spaces

from craft import utils

from .action_enum import ActionEnum
from .grid_objs import ObjType
from .matrix_env import MatrixEnv


class CraftEnv(MatrixEnv):

    def __init__(self, enable_render, init_blueprint_path, env_config):
        super().__init__(enable_render, init_blueprint_path, env_config)
        self.env_config = env_config
        self._ob_dim = 13
        self._init_obs_and_ac_dim()
        self.direction_to_binary = {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]}
        self.area_size = self._blackboard.area_size

        self.flag_pos_x = 0
        self.flag_pos_y = 0
        self.flag_pos_z = 0

        self._total_step = 0

        self.act_lift_cnt = 0
        self.act_drop_cnt = 0
        self.act_fold_cnt = 0
        self.act_unfo_cnt = 0
        self.lift_block_cnt = 0
        self.lift_slope_cnt = 0
        self.second_floor_cnt = 0
        self.third_floor_cnt = 0
        self.lift_flag_cnt = 0

        self.block_near_unfold_slope_body_cnt = 0
        self.building_complexity_score_max = 0
        self.block_near_unfold_slope_body_cnt_previous = -1
        self.building_complexity_score_list = []
        self.building_complexity_score_max_previous = -1
        self.second_floor_block_cnt = 0
        self.second_floor_block_cnt_previous = -1
        self.block_on_block_dict = {}
        self.block_on_block_dict_cnt = {}
        self.fold_on_block_dict = {}
        self.fold_on_block_dict_cnt = {}
        self.unfold_on_block_dict = {}
        self.unfold_on_block_dict_cnt = {}
        self.block_unfold_on_block_dict = {}
        self.block_unfold_on_block_dict_cnt = {}
        self.block_on_block_on_block_dict = {}
        self.block_on_block_on_block_dict_cnt = {}

    def _init_obs_and_ac_dim(self):
        enable_local_obs = self.env_config.get('enable_local_obs', False)
        blackboard = self._blackboard
        if enable_local_obs:
            max_free_num = self.env_config.get('local_max_free_num', 20)
            max_block_num = self.env_config.get('local_max_block_num',
                                                blackboard.block_num)
            max_slope_num = self.env_config.get('local_max_slope_num',
                                                blackboard.slope_num)

            obj_num = blackboard.block_num + blackboard.slope_num * 3 + 1 + 1
            self.ac_dim = 1 + max_block_num + \
                max_slope_num + max_free_num + len(ActionEnum)
            self.ob_dim = self._ob_dim * (obj_num + blackboard.smartcar_num + 1) + \
                self._ob_dim * (max_block_num + max_free_num + max_slope_num)
        else:
            obj_num = blackboard.block_num + blackboard.slope_num * 3 + 1 + 1
            self.ac_dim = len(ActionEnum)
            self.ob_dim = self._ob_dim * \
                (obj_num + blackboard.smartcar_num + 1)

    @property
    def observation_space(self):
        return spaces.Tuple([
            spaces.Tuple([
                spaces.Box(low=0,
                           high=np.inf,
                           shape=(self.ob_dim, ),
                           dtype=('float32')),
                spaces.Box(low=0,
                           high=1,
                           shape=(self.ac_dim, ),
                           dtype=('bool'))
            ]) for _ in range(self._blackboard.smartcar_num)
        ])

    @property
    def action_space(self):
        return spaces.Tuple([
            spaces.Discrete(self.ac_dim)
            for _ in range(self._blackboard.smartcar_num)
        ])

    def _get_global_obs(self, blackboard=None):
        if blackboard is None:
            blackboard = self._blackboard
        grid = blackboard.grid
        block_id = 0
        fold_slope_id = 0
        unfold_slope_body_id = 0
        unfold_slope_foot_id = 0
        """
        coordination, object class,
        x y z       , smartcar, foldslope, unfoldslopebody,
        unfoldslopefoot, block, flag, goal,

        object properties
        direction 0, direction 1(binary), is_lift(smartcar)

        3 + 7 + 3 = 13
        """
        obs_smartcar = np.zeros((blackboard.smartcar_num, self._ob_dim))
        obs_fold_slope = np.zeros((blackboard.slope_num, self._ob_dim))
        obs_unfold_slope_body = np.zeros((blackboard.slope_num, self._ob_dim))
        obs_unfold_slope_foot = np.zeros((blackboard.slope_num, self._ob_dim))
        obs_block = np.zeros((blackboard.block_num, self._ob_dim))

        for i, smartcar in enumerate(self.smartcars):
            obs_smartcar[i] = self.get_smartcar_vector(smartcar)
            x, y, z = smartcar.x, smartcar.y, smartcar.z
            smartcar.obj_under = grid[x][y][z - 1]

            if smartcar.is_lift:
                obj = smartcar.lift_obj
                if obj.type is ObjType.FoldedSlope:
                    obs_fold_slope[fold_slope_id] = self.get_object_vector(
                        obj.type, x, y, z, obj.yaw)

                    fold_slope_id += 1
                elif obj.type is ObjType.Block:
                    obs_block[block_id] = self.get_object_vector(
                        obj.type, x, y, z)

                    block_id += 1
                elif obj.type is ObjType.Flag:
                    obs_flag = self.get_object_vector(obj.type, x, y, z)

                    self.flag_pos_x = x
                    self.flag_pos_y = y
                    self.flag_pos_z = z

        self.block_near_unfold_slope_body_cnt = 0
        self.building_complexity_score_max = 0
        self.building_complexity_score_list = []
        self.second_floor_block_cnt = 0
        for i, j, k in itertools.product(range(self.area_size[0]),
                                         range(self.area_size[1]),
                                         range(1, self.area_size[2] + 1)):
            obj = grid[i][j][k]
            if obj.type is ObjType.FoldedSlope:
                obs_fold_slope[fold_slope_id] = self.get_object_vector(
                    obj.type, i, j, k, obj.yaw)
                fold_slope_id += 1
            elif obj.type is ObjType.UnfoldedSlopeBody:
                obs_unfold_slope_body[
                    unfold_slope_body_id] = self.get_object_vector(
                        obj.type, i, j, k, obj.yaw)
                unfold_slope_body_id += 1
            elif obj.type is ObjType.UnfoldedSlopeFoot:
                obs_unfold_slope_foot[
                    unfold_slope_foot_id] = self.get_object_vector(
                        obj.type, i, j, k, obj.yaw)
                unfold_slope_foot_id += 1
            elif obj.type is ObjType.Block:
                obs_block[block_id] = self.get_object_vector(obj.type, i, j, k)
                block_id += 1

                obj_on_it = grid[i][j][k + 1]
                xyz_key = str(i) + ',' + str(j) + ',' + str(k)
                if xyz_key in self.block_on_block_dict:
                    if obj_on_it.type is ObjType.Block:
                        self.block_on_block_dict[xyz_key] += 1
                    else:
                        self.block_on_block_dict[xyz_key] = 0
                else:
                    self.block_on_block_dict[str(i) + ',' + str(j) + ',' +
                                             str(k)] = 0

                if xyz_key in self.fold_on_block_dict:
                    if obj_on_it.type is ObjType.FoldedSlope:
                        self.fold_on_block_dict[xyz_key] += 1
                    else:
                        self.fold_on_block_dict[xyz_key] = 0
                else:
                    self.fold_on_block_dict[str(i) + ',' + str(j) + ',' +
                                            str(k)] = 0

                if xyz_key in self.unfold_on_block_dict:
                    if obj_on_it.type is ObjType.UnfoldedSlopeBody:
                        self.unfold_on_block_dict[xyz_key] += 1
                    else:
                        self.unfold_on_block_dict[xyz_key] = 0
                else:
                    self.unfold_on_block_dict[str(i) + ',' + str(j) + ',' +
                                              str(k)] = 0

                if xyz_key in self.block_unfold_on_block_dict:
                    if obj_on_it.type is ObjType.UnfoldedSlopeBody:
                        if (grid[i - 1][j][k + 1].type is ObjType.Block
                                and grid[i - 1][j][k + 1].near_unfold_slope_body)\
                                or (grid[i + 1][j][k + 1].type is ObjType.Block
                                    and grid[i + 1][j][k + 1].near_unfold_slope_body)\
                                or (grid[i][j - 1][k + 1].type is ObjType.Block
                                    and grid[i][j - 1][k + 1].near_unfold_slope_body)\
                                or (grid[i][j + 1][k + 1].type is ObjType.Block and
                                    grid[i][j + 1][k + 1].near_unfold_slope_body):
                            self.block_unfold_on_block_dict[xyz_key] += 1
                    else:
                        self.block_unfold_on_block_dict[xyz_key] = 0
                else:
                    self.block_unfold_on_block_dict[str(i) + ',' + str(j) +
                                                    ',' + str(k)] = 0

                if xyz_key in self.block_on_block_on_block_dict:
                    if (k == 1):
                        if (obj_on_it.type is ObjType.Block) and (
                                grid[i][j][k + 2].type is ObjType.Block):
                            self.block_on_block_on_block_dict[xyz_key] += 1
                    else:
                        self.block_on_block_on_block_dict[xyz_key] = 0
                else:
                    self.block_on_block_on_block_dict[str(i) + ',' + str(j) +
                                                      ',' + str(k)] = 0

                # the block is on the second floor and the four blocks under
                # should be in a row
                if (k == 2)\
                    and (grid[i][j][k - 1].type is ObjType.Block
                         and grid[i - 1][j][k - 1].type is ObjType.Block
                         and grid[i - 2][j][k - 1].type is ObjType.Block
                         and grid[i - 3][j][k - 1].type is ObjType.Block)\
                    or (grid[i][j][k - 1].type is ObjType.Block
                        and grid[i + 1][j][k - 1].type is ObjType.Block
                        and grid[i + 2][j][k - 1].type is ObjType.Block
                        and grid[i + 3][j][k - 1].type is ObjType.Block)\
                    or (grid[i][j][k - 1].type is ObjType.Block
                        and grid[i][j - 1][k - 1].type is ObjType.Block
                        and grid[i][j - 2][k - 1].type is ObjType.Block
                        and grid[i][j - 3][k - 1].type is ObjType.Block)\
                    or (grid[i][j][k - 1].type is ObjType.Block
                        and grid[i][j + 1][k - 1].type is ObjType.Block
                        and grid[i][j + 2][k - 1].type is ObjType.Block
                        and grid[i][j + 3][k - 1].type is ObjType.Block):
                    self.second_floor_block_cnt += 1

                if obj.near_unfold_slope_body:
                    self.block_near_unfold_slope_body_cnt += 1

        obs_flag = self.get_goal_vector()
        obs_goal = self.get_goal_vector()

        obs_units = np.concatenate([
            obs_fold_slope, obs_unfold_slope_body, obs_unfold_slope_foot,
            obs_block,
            obs_flag.reshape(1, self._ob_dim),
            obs_goal.reshape(1, self._ob_dim)
        ])

        obs_list = []
        for i in range(self._blackboard.smartcar_num):
            obs_smartcar_i = np.expand_dims(obs_smartcar[i], axis=0)
            obs_i = np.concatenate([obs_smartcar_i, obs_smartcar, obs_units])
            obs_list.append(
                (obs_i.flatten(), self.smartcars[i].get_action_mask()))

        return tuple(obs_list)

    def _get_local_obs(self, blackboard=None):
        if blackboard is None:
            blackboard = self._blackboard
        grid = blackboard.grid

        max_block_num = self.env_config.get('local_max_block_num',
                                            blackboard.block_num)
        max_slope_num = self.env_config.get('local_max_slope_num',
                                            blackboard.slope_num)
        max_free_num = self.env_config.get('local_max_free_num', 20)

        obs_list = []
        self.object_list = []

        for car in self.smartcars:
            blocks, slopes, frees = car.get_local_obj_lists()
            block_num = min(len(blocks), max_block_num)
            slope_num = min(len(slopes), max_slope_num)
            free_num = min(len(frees), max_free_num)

            car_obj_list = np.concatenate([
                np.array(blocks)[:block_num] if blocks else np.zeros([0, 3]) -
                1,
                np.zeros([max_block_num - block_num, 3]) - 1,
                np.array(slopes)[:slope_num] if slopes else np.zeros([0, 3]) -
                1,
                np.zeros([max_slope_num - slope_num, 3]) - 1,
                np.array(frees)[:free_num] if frees else np.zeros([0, 3]) - 1,
                np.zeros([max_free_num - free_num, 3]) - 1,
            ]).astype(int)
            self.object_list.append(car_obj_list)

            obs_block = np.zeros((max_block_num, self._ob_dim))
            obs_fold_slope = np.zeros((max_slope_num, self._ob_dim))
            obs_upper_free_space = np.zeros((max_free_num, self._ob_dim))

            # local obs
            for i in range(block_num):
                obs_block[i] = self.get_object_vector(ObjType.Block,
                                                      *blocks[i])
            for i in range(slope_num):
                x, y, z = slopes[i]
                obj_type, obj_yaw = grid[x][y][z].type, grid[x][y][z].yaw
                obs_fold_slope[i] = self.get_object_vector(
                    obj_type, *slopes[i], obj_yaw)
            for i in range(free_num):
                obs_upper_free_space[i] = self.get_object_vector(
                    ObjType.Air, *frees[i])
            car_obs = np.concatenate(
                [obs_block, obs_fold_slope, obs_upper_free_space])

            # high level action mask
            ac_mask = np.zeros((self.ac_dim, ), dtype=np.bool)
            if car.continue_last_cmd_cnt >= 0:
                ac_mask[0] = 1
            else:
                start_idx = 1
                ac_mask[start_idx:block_num + start_idx] = 1
                start_idx += max_block_num
                ac_mask[start_idx:slope_num + start_idx] = 1
                start_idx += max_slope_num
                ac_mask[start_idx:free_num + start_idx] = 1
                ac_mask[-len(ActionEnum):] = car.get_action_mask()

            obs_list.append((car_obs.flatten(), ac_mask))
        return tuple(obs_list)

    def get_obs(self):
        enable_local_obs = self.env_config.get('enable_local_obs', False)
        obs = self._get_global_obs()
        if not enable_local_obs:
            return obs

        local_obs = list(self._get_local_obs())
        obs = list(obs)
        for i, (obs_item, local_obs_item) in enumerate(zip(obs, local_obs)):
            car_obs, _ = obs_item
            local_car_obs, high_level_act_mask = local_obs_item
            car_obs = np.concatenate([car_obs, local_car_obs])
            obs[i] = (car_obs, high_level_act_mask)
        return tuple(obs)

    def step(self, action):
        enable_local_obs = self.env_config.get('enable_local_obs', False)
        if not enable_local_obs:
            return super().step(action)

        action_order = np.arange(self._blackboard.smartcar_num)
        np.random.shuffle(action_order)

        if self.work_mode == 2:
            try:
                action, action_order = self.record_action[
                    self._step], self.record_action_order[self._step]
            except IndexError:
                print("record actions reach limit.")
                return None, 0, False, {}

        # execute high level action
        # high level action: 1(action_helper) + max_block_num + max_slope_num +
        # max_free_num + len(ActionEnum)
        for i in action_order:
            car_act = action[i]
            # 1. action helper: call car.continue_action()
            if car_act == 0:
                self.smartcars[i].continue_action()
            # 2. exec action enum: call car.step()
            elif car_act >= self.ac_dim - len(ActionEnum):
                a = car_act - (self.ac_dim - len(ActionEnum))
                self.smartcars[i].step(a)
            # 3. select from self.object_list: call car.plan()
            elif car_act >= 1 and car_act < self.ac_dim - len(ActionEnum):
                selected_obj = self.object_list[i][car_act - 1]
                self.smartcars[i].plan(*selected_obj)

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

    def get_flag_pos(self, blackboard=None):
        return self.flag_pos_x, self.flag_pos_y, self.flag_pos_z

    def get_block_near_unfold_slope_body_cnt(self, blackboard=None):
        return self.block_near_unfold_slope_body_cnt

    def get_building_complexity_score_max(self, blackboard=None):
        return self.building_complexity_score_max

    def get_second_floor_block_cnt(self, blackboard=None):
        return self.second_floor_block_cnt

    def get_block_on_block_dict(self, blackboard=None):
        return self.block_on_block_dict

    def reset_block_on_block_dict(self, blackboard=None):
        self.block_on_block_dict = {}

    def get_fold_on_block_dict(self, blackboard=None):
        return self.fold_on_block_dict

    def reset_fold_on_block_dict(self, blackboard=None):
        self.fold_on_block_dict = {}

    def get_unfold_on_block_dict(self, blackboard=None):
        return self.unfold_on_block_dict

    def reset_unfold_on_block_dict(self, blackboard=None):
        self.unfold_on_block_dict = {}

    def get_block_unfold_on_block_dict(self, blackboard=None):
        return self.block_unfold_on_block_dict

    def reset_block_unfold_on_block_dict(self, blackboard=None):
        self.block_unfold_on_block_dict = {}

    def get_block_on_block_on_block_dict(self, blackboard=None):
        return self.block_on_block_on_block_dict

    def reset_block_on_block_on_block_dict(self, blackboard=None):
        self.block_on_block_on_block_dict = {}

    def reset(self):
        obs = super().reset()
        self.act_lift_cnt = 0
        self.act_drop_cnt = 0
        self.act_fold_cnt = 0
        self.act_unfo_cnt = 0
        self.lift_block_cnt = 0
        self.lift_slope_cnt = 0
        self.second_floor_cnt = 0
        self.third_floor_cnt = 0
        self.lift_flag_cnt = 0
        self.reachable_space = 0
        self.reachable_space_size_previous = -1
        self.block_near_unfold_slope_body_cnt_previous = -1
        self.building_complexity_score_max = -1
        self.building_complexity_score_max_previous = -1
        self.second_floor_block_cnt_previous = -1
        self.reset_block_on_block_dict()
        self.reset_fold_on_block_dict()
        self.reset_unfold_on_block_dict()
        self.reset_block_unfold_on_block_dict()
        self.reset_block_on_block_on_block_dict()
        self.block_on_block_dict = {}
        self.block_on_block_dict_cnt = {}
        self.fold_on_block_dict = {}
        self.fold_on_block_dict_cnt = {}
        self.unfold_on_block_dict = {}
        self.unfold_on_block_dict_cnt = {}
        self.block_unfold_on_block_dict = {}
        self.block_unfold_on_block_dict_cnt = {}
        self.block_on_block_on_block_dict = {}
        self.block_on_block_on_block_dict_cnt = {}
        return obs

    def get_object_vector(self, obj_type, x, y, z, yaw=None):
        """
        observation representation ( binary vector/embedding )
        x, y, z,
        is_smartcar, is_foldedslope, is_unfoldedslopebody,
        is_unfoldedslopefoot, is_block, is_flag, is_goal
        direction0, direction1, is_lift(smartcar)
        _obs_dim = 3 + 7 + 3 = 13
        """
        if obj_type is ObjType.FoldedSlope:
            dir0, dir1 = self.direction_to_binary[yaw]
            return np.array([x, y, z, 0, 1, 0, 0, 0, 0, 0, dir0, dir1, 0])
        if obj_type is ObjType.UnfoldedSlopeBody:
            dir0, dir1 = self.direction_to_binary[yaw]
            return np.array([x, y, z, 0, 0, 1, 0, 0, 0, 0, dir0, dir1, 0])
        if obj_type is ObjType.UnfoldedSlopeFoot:
            dir0, dir1 = self.direction_to_binary[yaw]
            return np.array([x, y, z, 0, 0, 0, 1, 0, 0, 0, dir0, dir1, 0])
        if obj_type is ObjType.Block:
            return np.array([x, y, z, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        if obj_type is ObjType.Flag:
            return np.array([x, y, z, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        if obj_type is ObjType.Air:
            return np.array([x, y, z, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        raise TypeError

    def get_goal_vector(self):
        """
        observation representation ( binary vector/embedding )
        x, y, z,
        is_smartcar, is_foldedslope, is_unfoldedslopebody,
        is_unfoldedslopefoot, is_block, is_flag, is_goal
        direction0, direction1, is_lift(smartcar)
        3 + 7 + 3 = 13
        """
        goal = self._blackboard.goal
        return np.array([goal.x, goal.y, goal.z, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])

    def get_smartcar_vector(self, car):
        """
        observation representation ( binary vector/embedding )
        x, y, z,
        is_smartcar, is_foldedslope, is_unfoldedslopebody,
        is_unfoldedslopefoot, is_block, is_flag, is_goal
        direction0, direction1, is_lift(smartcar)
        3 + 7 + 3 = 13
        """
        dir0, dir1 = self.direction_to_binary[car.yaw]
        return np.array([
            car.x, car.y, car.z, 1, 0, 0, 0, 0, 0, 0, dir0, dir1, car.is_lift
        ])

    def _is_done(self, blackboard=None):
        """Indicates whether or not the episode is done
        ( the robot has fallen for example)."""
        if blackboard is None:
            blackboard = self._blackboard

        if self._step > self.env_config['max_steps']:
            return True
        else:
            return False

    def _compute_reward(self, blackboard=None):
        """Calculate the reward."""
        reward = 0
        if self.act_lift_cnt < self.env_config['reward_cnt']:
            if any(smartcar.act_lift and smartcar.is_lift is True
                   and smartcar.lift_obj.type is ObjType.FoldedSlope
                   for smartcar in self.smartcars):
                reward += self.env_config['act_lift']
                self.act_lift_cnt += 1
                for smartcar in self.smartcars:
                    smartcar.act_lift = False

        if self.act_drop_cnt < self.env_config['reward_cnt']:
            if any(smartcar.act_drop and smartcar.is_lift is False
                   for smartcar in self.smartcars):
                reward += self.env_config['act_drop']
                self.act_drop_cnt += 1
                for smartcar in self.smartcars:
                    smartcar.act_drop = False

        if self.act_fold_cnt < self.env_config['reward_cnt']:
            if any(smartcar.act_fold for smartcar in self.smartcars):
                reward += self.env_config['act_fold']
                self.act_fold_cnt += 1
                for smartcar in self.smartcars:
                    smartcar.act_fold = False

        if self.act_unfo_cnt < self.env_config['reward_cnt']:
            if any(smartcar.act_unfo for smartcar in self.smartcars):
                reward += self.env_config['act_unfo']
                self.act_unfo_cnt += 1
                for smartcar in self.smartcars:
                    smartcar.act_unfo = False

        if self.lift_block_cnt < self.env_config['reward_cnt']:
            if any(smartcar.is_lift and smartcar.lift_obj.type is ObjType.Block
                   for smartcar in self.smartcars):
                reward += self.env_config['lift_block']
                self.lift_block_cnt += 1

        if self.lift_slope_cnt < self.env_config['reward_cnt']:
            if any(smartcar.is_lift
                   and smartcar.lift_obj.type is ObjType.FoldedSlope
                   for smartcar in self.smartcars):
                reward += self.env_config['lift_slope']
                self.lift_slope_cnt += 1

        if self.second_floor_cnt < self.env_config['reward_cnt']:
            for smartcar in self.smartcars:
                if (smartcar.z
                        == 2) and smartcar.obj_under.type is ObjType.Block:
                    reward += self.env_config['second_floor']
                    self.second_floor_cnt += 1

        if self.third_floor_cnt < self.env_config['reward_cnt']:
            for smartcar in self.smartcars:
                if (smartcar.z
                        == 3) and smartcar.obj_under.type is ObjType.Block:
                    reward += self.env_config['third_floor']
                    self.third_floor_cnt += 1

        if self.lift_flag_cnt < self.env_config['reward_cnt']:
            if any(smartcar.is_lift and smartcar.lift_obj.type is ObjType.Flag
                   for smartcar in self.smartcars):
                reward += self.env_config['lift_flag']
                self.lift_flag_cnt += 1

        self.flag_pos_x, self.flag_pos_y, self.flag_pos_z = self.get_flag_pos()
        if self.flag_pos_x == self._blackboard.goal.x \
                and self.flag_pos_y == self._blackboard.goal.y \
                and self.flag_pos_z == self._blackboard.goal.z:
            reward += self.env_config['reach_goal']

        reward -= self.env_config['step_penalty']

        # build block reward:
        self.block_near_unfold_slope_body_cnt = \
            self.get_block_near_unfold_slope_body_cnt(
            )
        if self.block_near_unfold_slope_body_cnt_previous < 0:
            self.block_near_unfold_slope_body_cnt_previous = \
                self.block_near_unfold_slope_body_cnt
        else:
            block_near_unfold_diff = self.block_near_unfold_slope_body_cnt - \
                self.block_near_unfold_slope_body_cnt_previous
            reward += block_near_unfold_diff * \
                self.env_config['block_near_unfold']
            self.block_near_unfold_slope_body_cnt_previous = \
                self.block_near_unfold_slope_body_cnt

        self.building_complexity_score_max = \
            self.get_building_complexity_score_max(
            )
        if self.building_complexity_score_max_previous < 0:
            self.building_complexity_score_max_previous = \
                self.building_complexity_score_max
        else:
            building_complexity_diff = self.building_complexity_score_max - \
                self.building_complexity_score_max_previous
            reward += building_complexity_diff * \
                self.env_config['building_complexity']
            self.building_complexity_score_max_previous = \
                self.building_complexity_score_max

        self.second_floor_block_cnt = self.get_second_floor_block_cnt()
        if self.second_floor_block_cnt_previous < 0:
            self.second_floor_block_cnt_previous = self.second_floor_block_cnt
        else:
            second_floor_block_cnt_diff = self.second_floor_block_cnt - \
                self.second_floor_block_cnt_previous
            reward += second_floor_block_cnt_diff * \
                self.env_config['second_floor_block']
            self.second_floor_block_cnt_previous = self.second_floor_block_cnt

        self.block_on_block_dict = self.get_block_on_block_dict()
        for key in self.block_on_block_dict:
            if not (key in self.block_on_block_dict_cnt):
                self.block_on_block_dict_cnt[key] = 0
            if self.block_on_block_dict[key] == 1 and self._step > 1:
                if self.block_on_block_dict_cnt[key] < self.env_config[
                        'reward_cnt']:
                    reward += self.env_config['block_on_block']
                self.block_on_block_dict_cnt[key] += 1

        self.fold_on_block_dict = self.get_fold_on_block_dict()
        for key in self.fold_on_block_dict:
            if not (key in self.fold_on_block_dict_cnt):
                self.fold_on_block_dict_cnt[key] = 0
            if self.fold_on_block_dict[key] == 1 and self._step > 1:
                if self.fold_on_block_dict_cnt[key] < self.env_config[
                        'reward_cnt']:
                    reward += self.env_config['fold_on_block']
                self.fold_on_block_dict_cnt[key] += 1

        self.unfold_on_block_dict = self.get_unfold_on_block_dict()
        for key in self.unfold_on_block_dict:
            if not (key in self.unfold_on_block_dict_cnt):
                self.unfold_on_block_dict_cnt[key] = 0
            if self.unfold_on_block_dict[key] == 1 and self._step > 1:
                if self.unfold_on_block_dict_cnt[key] < self.env_config[
                        'reward_cnt']:
                    reward += self.env_config['unfold_on_block']
                self.unfold_on_block_dict_cnt[key] += 1

        self.block_unfold_on_block_dict = self.get_block_unfold_on_block_dict()
        for key in self.block_unfold_on_block_dict:
            if not (key in self.block_unfold_on_block_dict_cnt):
                self.block_unfold_on_block_dict_cnt[key] = 0
            if self.block_unfold_on_block_dict[key] == 1 and self._step > 1:
                if self.block_unfold_on_block_dict_cnt[key] < self.env_config[
                        'reward_cnt']:
                    reward += self.env_config['block_unfold_on_block']
                self.block_unfold_on_block_dict_cnt[key] += 1

        self.block_on_block_on_block_dict = \
            self.get_block_on_block_on_block_dict(
            )
        for key in self.block_on_block_on_block_dict:
            if not (key in self.block_on_block_on_block_dict_cnt):
                self.block_on_block_on_block_dict_cnt[key] = 0
            if self.block_on_block_on_block_dict[key] == 1 and self._step > 1:
                if self.block_on_block_on_block_dict_cnt[
                        key] < self.env_config['reward_cnt']:
                    reward += self.env_config['block_on_block_on_block']
                self.block_on_block_on_block_dict_cnt[key] += 1
        return reward
