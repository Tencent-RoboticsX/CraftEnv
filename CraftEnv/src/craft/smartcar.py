import numpy as np

from craft import utils

from .action_enum import ACTION_ARG, ActionEnum
from .blackboard import Point
from .grid_objs import (Air, FoldedSlope, FoldedSlopeGear, ObjType,
                        UnfoldedSlopeBody, UnfoldedSlopeFoot)
from .planner import AStarPlanner


class Smartcar:

    def __init__(self,
                 blackboard,
                 id_,
                 enable_action_helper=True,
                 agent_view_size=7,
                 search_depth=10):
        super().__init__()
        self._blackboard = blackboard
        self.area_size = blackboard.area_size
        self._id = id_
        self.x = 0
        self.y = 0
        self.z = 1
        self.yaw = 0
        self.is_lift = False
        self.lift_obj = None

        # 0 -> moving in free space
        # 1 -> UnfoldedSlopeFoot
        # 2 -> UnfoldedSlopeBody
        self.moving_over_slope = 0
        self.enable_action_helper = enable_action_helper
        self.view_size = agent_view_size
        self.continue_last_cmd_cnt = -1
        self.action_list = []

        self.action_func = {
            ActionEnum.LIFT: self.lift,
            ActionEnum.DROP: self.drop,
            ActionEnum.FOLD: self.fold,
            ActionEnum.UNFOLD: self.unfold,
            ActionEnum.ROTATE_LEFT: self.rotate,
            ActionEnum.ROTATE_RIGHT: self.rotate,
            ActionEnum.MOVE_FORWARD: self.move_action,
            ActionEnum.MOVE_BACK: self.move_action,
            ActionEnum.MOVE_LEFT: self.move_action,
            ActionEnum.MOVE_RIGHT: self.move_action,
            ActionEnum.STOP: self.move_stop,
            ActionEnum.MOVE_FORWARD_LEFT: self.move_action,
            ActionEnum.MOVE_FORWARD_RIGHT: self.move_action,
            ActionEnum.MOVE_BACK_LEFT: self.move_action,
            ActionEnum.MOVE_BACK_RIGHT: self.move_action,
        }
        self.action_mask = []

        self.act_lift = False
        self.act_drop = False
        self.act_fold = False
        self.act_unfo = False
        self.obj_under = None

        self.planner = AStarPlanner(blackboard, search_depth=search_depth)

    def set_pose(self, x, y, z, yaw):
        assert x >= 0 and x < self.area_size[0]
        assert y >= 0 and y < self.area_size[1]
        assert z >= 1 and z <= self.area_size[2]
        assert yaw >= 0 and yaw <= 3
        self.x, self.y, self.z, self.yaw = x, y, z, yaw

    def reset(self):
        self.template = self._blackboard.template
        self.act_lift = False
        self.act_drop = False
        self.act_fold = False
        self.act_unfo = False
        self.obj_under = None
        flag = False
        info_dict = {}
        try:
            for i in self.template["smartcar"]:
                if self._id == i["id"]:
                    info_dict = i
                    flag = True
        except KeyError as e:
            print("KeyError, ", e)
            pass
        if flag:
            self.x = info_dict["x"]
            self.y = info_dict["y"]
            self.z = info_dict["z"]
            self.yaw = info_dict["yaw"]
            self._blackboard.spawn_point_set.add(Point(self.x, self.y, self.z))
        else:
            # random spawn
            p, direction = self._blackboard.random_spawn_obj("smartcar")
            self._blackboard.spawn_point_set.add(p)
            self.x = p.x
            self.y = p.y
            self.z = p.z
            self.yaw = direction
        self.is_lift = False
        self.lift_obj = None
        self._blackboard.grid[self.x][self.y][self.z - 1].obj_on_it = self._id

        self.planner.reset()

    def goal_from_action(self, a):
        move_dir = ACTION_ARG[a]
        goal_x = self.x + move_dir[0]
        goal_y = self.y + move_dir[1]
        goal_z = self.z
        if self.moving_over_slope != 0:
            obj = self._blackboard.grid[goal_x][goal_y][self.z]
            blow_obj = self._blackboard.grid[goal_x][goal_y][self.z - 1]
            if obj.type is ObjType.UnfoldedSlopeBody and \
                    self.moving_over_slope == 1:
                goal_z = self.z + 1
            elif blow_obj.type is ObjType.UnfoldedSlopeFoot and \
                    self.moving_over_slope == 2:
                goal_z = self.z - 1

        return (goal_x, goal_y, goal_z)

    def get_action_mask(self):
        action_mask_proxy = self._blackboard.action_mask_proxy
        action_mask = action_mask_proxy.calc_mask(self._id)

        for a in ActionEnum:
            if utils.is_move_action(a):
                start = (self.x, self.y, self.z, self.yaw, self.is_lift,
                         self.lift_obj, self.moving_over_slope)
                goal = self.goal_from_action(a)
                if self.planner.plan(self._id, *start, *goal, verbose=False):
                    action_mask[a] = True
        return action_mask

    def step(self, a):
        a = int(a)
        if self.continue_last_cmd_cnt >= 0:
            self.continue_action()
            return

        if self.enable_action_helper and utils.is_move_action(a):
            start = (self.x, self.y, self.z, self.yaw, self.is_lift,
                     self.lift_obj, self.moving_over_slope)
            goal = self.goal_from_action(a)
            path = self.planner.plan(self._id, *start, *goal, verbose=False)
            self.action_list = self.planner.get_actions(path)
            self.action_list.reverse()
            self.continue_last_cmd_cnt = len(self.action_list) - 1
            self.continue_action()
        else:
            action_mask = self._blackboard.action_mask_proxy.calc_mask(
                self._id)
            if action_mask[a]:
                self.action_func[a](ACTION_ARG[a])

    def continue_action(self):
        assert self.continue_last_cmd_cnt >= 0
        action = self.action_list[self.continue_last_cmd_cnt]
        action_mask = self._blackboard.action_mask_proxy.calc_mask(self._id)
        if action_mask[action]:
            self.action_func[action](ACTION_ARG[action])
            self.continue_last_cmd_cnt -= 1
            self.action_list.pop()
        else:
            self.continue_last_cmd_cnt = -1
            self.action_list.clear()

    def plan(self, x, y, z):
        if self.continue_last_cmd_cnt >= 0:
            self.continue_action()
        else:
            goal = (x, y, z)
            start = (self.x, self.y, self.z, self.yaw, self.is_lift,
                     self.lift_obj, self.moving_over_slope)
            path = self.planner.plan(self._id, *start, *goal, verbose=False)
            self.action_list = self.planner.get_actions(path)
            self.action_list.reverse()
            self.continue_last_cmd_cnt = len(self.action_list) - 1

            self.continue_action()

    def lift(self, _):
        obj = self._blackboard.grid[self.x][self.y][self.z]
        self.is_lift = True
        self.lift_obj = obj
        self._blackboard.grid[self.x][self.y][self.z] = Air()
        if obj.type is ObjType.FoldedSlope:
            n_x, n_y = utils.next_step(self.x, self.y, obj.yaw * np.pi / 2)
            self._blackboard.grid[n_x][n_y][self.z] = Air()

        self.lift_obj.near_unfold_slope_body = False
        self.lift_obj.near_blow_unfold_slope_foot = False
        self._blackboard.grid[self.x][self.y][self.z - 1].obj_on_it = self._id
        self.act_lift = True

    def drop(self, _):
        self._blackboard.grid[self.x][self.y][self.z] = self.lift_obj
        if self.lift_obj.type is ObjType.FoldedSlope:
            n_x, n_y = utils.next_step(self.x, self.y,
                                       self.lift_obj.yaw * np.pi / 2)
            self._blackboard.grid[n_x][n_y][self.z] = FoldedSlopeGear(
                self.lift_obj.yaw)

        # When the block is put down, see if it is against the
        # UnfoldedSlopeBody, and then set near_unfold_slope_body
        if self.lift_obj.type is ObjType.Block:
            for i in range(4):
                n_x, n_y = utils.next_step(self.x, self.y, i * np.pi / 2)
                n_obj = self._blackboard.grid[n_x][n_y][self.z]
                if n_obj.type is ObjType.UnfoldedSlopeBody and i == n_obj.yaw:
                    self.lift_obj.near_unfold_slope_body = True
                    break
            for i in range(4):
                n_x, n_y = utils.next_step(self.x, self.y, i * np.pi / 2)
                n_above = self._blackboard.grid[n_x][n_y][self.z + 1]
                if n_above.type is ObjType.UnfoldedSlopeFoot and abs(
                        i - n_above.yaw) == 2:
                    self.lift_obj.near_blow_unfold_slope_foot = True
                    break

        self._blackboard.grid[self.x][self.y][self.z - 1].obj_on_it = self._id

        self.is_lift = False
        self.lift_obj = None
        self.act_drop = True

    def fold(self, _):
        slope_direction = (self.yaw + 2) % 4
        self._blackboard.grid[self.x][self.y][self.z] = FoldedSlope(
            slope_direction)
        n_x, n_y = utils.next_step(self.x, self.y, slope_direction * np.pi / 2)
        self._blackboard.grid[n_x][n_y][self.z] = FoldedSlopeGear(
            slope_direction)
        self._blackboard.grid[n_x][n_y][self.z - 1].obj_on_it = -1
        self.set_obj_config(False)
        self.act_fold = True

    def unfold(self, _):
        slope_direction = (self.yaw + 2) % 4
        self._blackboard.grid[self.x][self.y][self.z] = UnfoldedSlopeBody(
            slope_direction)
        n_x, n_y = utils.next_step(self.x, self.y, slope_direction * np.pi / 2)
        self._blackboard.grid[n_x][n_y][self.z] = UnfoldedSlopeFoot(
            slope_direction)
        self._blackboard.grid[n_x][n_y][self.z - 1].obj_on_it = -2
        self.set_obj_config(True)
        self.act_unfo = True

    def set_obj_config(self, state):
        n_x, n_y = utils.next_step(self.x, self.y, self.yaw * np.pi / 2)
        self._blackboard.grid[n_x][n_y][self.z].near_unfold_slope_body = state
        slope_direction = (self.yaw + 2) % 4
        pre_x, pre_y = utils.next_step(self.x, self.y,
                                       slope_direction * np.pi / 2)
        ppre_x, ppre_y = utils.next_step(pre_x, pre_y,
                                         slope_direction * np.pi / 2)
        self._blackboard.grid[ppre_x][ppre_y][
            self.z - 1].near_blow_unfold_slope_foot = state

    def rotate(self, rotate_dir):
        self.yaw = (self.yaw + rotate_dir) % 4
        if self.is_lift and self.lift_obj.type is ObjType.FoldedSlope:
            self.lift_obj.yaw = (self.lift_obj.yaw + rotate_dir) % 4

    def move_stop(self, _):
        return

    def move_action(self, a):
        self._blackboard.grid[self.x][self.y][self.z - 1].obj_on_it = (
            -1 if
            self._blackboard.grid[self.x][self.y][self.z].type is ObjType.Air
            else -2)

        self.x += a[0]
        self.y += a[1]
        obj = self._blackboard.grid[self.x][self.y][self.z]
        blow_obj = self._blackboard.grid[self.x][self.y][self.z - 1]

        if obj.type is ObjType.UnfoldedSlopeFoot and \
                self.moving_over_slope == 0:
            self.moving_over_slope = 1
        elif obj.type is ObjType.UnfoldedSlopeBody and \
                self.moving_over_slope == 1:
            self.moving_over_slope = 2
            self.z += 1
        elif blow_obj.type is ObjType.UnfoldedSlopeBody \
                and self.moving_over_slope == 0:
            self.moving_over_slope = 2
        elif blow_obj.type is ObjType.UnfoldedSlopeFoot and \
                self.moving_over_slope == 2:
            self.moving_over_slope = 1
            self.z -= 1
        else:
            self.moving_over_slope = 0

        self._blackboard.grid[self.x][self.y][self.z - 1].obj_on_it = self._id

    def get_local_obj_lists(self):
        state = (self._id, self.x, self.y, self.z, self.yaw,
                 self.moving_over_slope)

        space = self._blackboard.bfs.search(*state,
                                            visualize=False,
                                            view_size=self.view_size,
                                            ignore_cars=False)

        blocks, slopes, drops = [], [], []

        grid = self._blackboard.grid
        obj_indices = space.nonzero()
        for x, y, z in zip(*obj_indices):
            obj = grid[x][y][z]
            if obj.type is ObjType.Block:
                blocks.append((x, y, z))
            if obj.type in (ObjType.FoldedSlope, ObjType.UnfoldedSlopeBody):
                slopes.append((x, y, z))
            if z > 1 and obj.type is ObjType.Air:
                drops.append((x, y, z))

        return blocks, slopes, drops
