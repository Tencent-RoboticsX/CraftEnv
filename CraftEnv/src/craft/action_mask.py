import numpy as np

from craft import utils

from .action_enum import ACTION_ARG, ActionEnum
from .grid_objs import ObjType


class ActionMask:
    def __init__(self, blackboard):
        self._blackboard = blackboard

        self.action_mask_func = {
            ActionEnum.LIFT: self.lift_mask,
            ActionEnum.DROP: self.drop_mask,
            ActionEnum.FOLD: self.fold_mask,
            ActionEnum.UNFOLD: self.unfold_mask,
            ActionEnum.ROTATE_LEFT: self.rotate_mask,
            ActionEnum.ROTATE_RIGHT: self.rotate_mask,
            ActionEnum.MOVE_FORWARD: self.move_action_mask,
            ActionEnum.MOVE_BACK: self.move_action_mask,
            ActionEnum.MOVE_LEFT: self.move_action_mask,
            ActionEnum.MOVE_RIGHT: self.move_action_mask,
            ActionEnum.STOP: self.stop_action_mask,
            ActionEnum.MOVE_FORWARD_LEFT: self.move_action_mask,
            ActionEnum.MOVE_FORWARD_RIGHT: self.move_action_mask,
            ActionEnum.MOVE_BACK_LEFT: self.move_action_mask,
            ActionEnum.MOVE_BACK_RIGHT: self.move_action_mask,
        }

    def get_smartcar_state(self, id_):
        x = self._blackboard.smartcars[id_].x
        y = self._blackboard.smartcars[id_].y
        z = self._blackboard.smartcars[id_].z
        yaw = self._blackboard.smartcars[id_].yaw
        is_lift = self._blackboard.smartcars[id_].is_lift
        lift_obj = self._blackboard.smartcars[id_].lift_obj
        moving_over_slope = self._blackboard.smartcars[id_].moving_over_slope
        return [x, y, z, yaw, is_lift, lift_obj, moving_over_slope]

    def calc_mask(self, id_):
        self._action_mask = np.zeros(len(ActionEnum), dtype=np.bool)
        self._action_mask[ActionEnum.STOP] = True

        for a in ActionEnum:
            self._action_mask[a] = self.action_mask_func[a](ACTION_ARG[a], id_)

        return self._action_mask

    def lift_mask(self, _, id_):
        [x, y, z, _, is_lift, _, moving_over_slope] = self.get_smartcar_state(id_)
        obj = self._blackboard.grid[x][y][z]
        if moving_over_slope != 0:
            return False
        return obj.can_lift and not is_lift and obj.obj_on_it == -1

    def drop_mask(self, _, id_):
        [_, _, _, _, is_lift, _, moving_over_slope] = self.get_smartcar_state(id_)
        if moving_over_slope != 0:
            return False
        return is_lift

    def fold_mask(self, _, id_):
        yaw_to_dir = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        def next_step(x, y, yaw):
            dx, dy = yaw_to_dir[yaw]
            return (x + dx, y + dy)

        [x, y, z, _, _, _, moving_over_slope] = self.get_smartcar_state(id_)
        obj = self._blackboard.grid[x][y][z]
        if moving_over_slope != 0 or obj.type is not ObjType.UnfoldedSlopeBody:
            return False
        if obj.obj_on_it != -1:
            return False
        foot_x, foot_y = next_step(x, y, obj.yaw)
        if self._blackboard.grid[foot_x][foot_y][z - 1].obj_on_it >= 0:
            return False
        return True

    def unfold_mask(self, _, id_):
        [x, y, z, _, is_lift, _, moving_over_slope] = self.get_smartcar_state(id_)
        obj = self._blackboard.grid[x][y][z]
        if moving_over_slope != 0:
            return False
        if is_lift or obj.type is not ObjType.FoldedSlope:
            return False

        next_x, next_y = utils.next_step(x, y, obj.yaw * np.pi / 2)
        next_obj = self._blackboard.grid[next_x][next_y][z]
        blow_obj = self._blackboard.grid[x][y][z - 1]
        next_blow_obj = self._blackboard.grid[next_x][next_y][z - 1]

        if (
            next_obj.type in (ObjType.Air, ObjType.FoldedSlopeGear)
            and next_blow_obj.can_stand
            and not next_blow_obj.near_unfold_slope_body
            and not blow_obj.near_unfold_slope_body
        ):
            return True

        return False

    def rotate_mask(self, rotate_dir, id_, kw=None, ignore_cars=False):
        yaw_to_dir = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        def next_step(x, y, yaw):
            dx, dy = yaw_to_dir[yaw]
            return (x + dx, y + dy)

        if kw:
            [x, y, z, _, is_lift, lift_obj, moving_over_slope] = \
                kw["x"], kw["y"], kw["z"], kw["yaw"], kw["is_lift"], kw["lift_obj"], kw["moving_over_slope"]
        else:
            [x, y, z, _, is_lift, lift_obj, moving_over_slope] = self.get_smartcar_state(id_)

        grid = self._blackboard.grid
        obj = grid[x][y][z]

        # ===== Not Lifting Object =====
        if moving_over_slope != 0:
            return False

        if not is_lift:
            return obj.type not in (ObjType.FoldedSlope, ObjType.UnfoldedSlopeBody)

        # ===== Lifting Object =====

        # collision between lift_obj and gear on grid
        for dx, dy, dir in [(1, 0, 0), (0, 1, 1), (-1, 0, 2), (0, -1, 3)]:
            neigh_obj = grid[x + dx][y + dy][z]
            if neigh_obj.type is ObjType.FoldedSlopeGear and abs(neigh_obj.yaw - dir) != 2:
                return False

        if not ignore_cars:
            smartcars = self._blackboard.smartcars
            lifted_obj_pos = [
                (car.x, car.y, car.z)
                for i, car in enumerate(smartcars) if car.is_lift and i != id_
            ]
            # collision between orthogonal lift_obj neighbors
            for obj_pos in lifted_obj_pos:
                if any(obj_pos == neigh_pos for neigh_pos in
                       [(x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z)]):
                    return False

            gear_pose = [
                (*next_step(car.x, car.y, car.lift_obj.yaw), car.z, car.lift_obj.yaw)
                for i, car in enumerate(smartcars)
                if car.is_lift and car.lift_obj.type is ObjType.FoldedSlope and i != id_
            ]
            # collision between orthogonal lifted gear neighbors
            for g_pose in gear_pose:
                if any(g_pose[:3] == neigh_pos and abs(g_pose[3] - dir) != 2 for neigh_pos, dir in
                       zip([(x + 1, y, z), (x, y + 1, z), (x - 1, y, z), (x, y - 1, z)], [0, 1, 2, 3])
                       ):
                    return False

        if lift_obj.type is ObjType.FoldedSlope:
            # check the diagonal neighbor swapped by lifted gear
            gear_yaw = lift_obj.yaw
            next_gear_yaw = (gear_yaw + rotate_dir) % 4

            # (0 -> 1) case
            if gear_yaw + next_gear_yaw == 1:
                dx, dy = 1, 1
            # (2 -> 3) case
            elif gear_yaw + next_gear_yaw == 5:
                dx, dy = -1, -1
            # (3 -> 0) case
            elif gear_yaw + next_gear_yaw == 3 and gear_yaw * next_gear_yaw == 0:
                dx, dy = 1, -1
            # (1 -> 2) case
            else:
                dx, dy = -1, 1

            diag_x, diag_y = x + dx, y + dy
            if not grid[diag_x][diag_y][z].type in (ObjType.Air, ObjType.FoldedSlopeGear):
                return False
            if not ignore_cars and any((diag_x, diag_y, z) == obj_pos for obj_pos in lifted_obj_pos):
                return False

        #    check orthogonal grid neighbors
        return all([grid[x + 1][y][z].type in (ObjType.Air, ObjType.FoldedSlopeGear),
                    grid[x - 1][y][z].type in (ObjType.Air, ObjType.FoldedSlopeGear),
                    grid[x][y + 1][z].type in (ObjType.Air, ObjType.FoldedSlopeGear),
                    grid[x][y - 1][z].type in (ObjType.Air, ObjType.FoldedSlopeGear)])

    def move_action_mask(self, move_dir, id_, kw=None, ignore_cars=False):
        """
        move_mask is designed to validate actions in two stages:
        1. state valid:      validate the movement (move_dir) at the current state (restricted by yaw constraint etc.)
        2. transition valid: validate transition to the next position, discuss conditions distinguished by next_obj.type
        """
        yaw_to_dir = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        def next_step(x, y, yaw):
            dx, dy = yaw_to_dir[yaw]
            return (x + dx, y + dy)

        if kw:
            [x, y, z, yaw, is_lift, lift_obj, moving_over_slope] = \
                kw["x"], kw["y"], kw["z"], kw["yaw"], kw["is_lift"], kw["lift_obj"], kw["moving_over_slope"]
        else:
            [x, y, z, yaw, is_lift, lift_obj, moving_over_slope] = self.get_smartcar_state(id_)

        grid = self._blackboard.grid

        obj = grid[x][y][z]
        next_x, next_y = x + move_dir[0], y + move_dir[1]
        next_obj = grid[next_x][next_y][z]
        next_blow_obj = grid[next_x][next_y][z - 1]

        # Disable diagonal movement
        if move_dir[0] != 0 and move_dir[1] != 0:
            return False

        # ======= State Validation =======
        forward_n_x, forward_n_y = next_step(x, y, yaw)
        is_move_heading_forward = forward_n_x == next_x and forward_n_y == next_y

        # Upon Slope
        if moving_over_slope == 1:
            if not is_move_heading_forward:
                return False
            if next_obj.type is ObjType.UnfoldedSlopeBody:
                return True
        if moving_over_slope == 2:
            if not is_move_heading_forward:
                return False
            if next_blow_obj.type is ObjType.UnfoldedSlopeFoot:
                return True

        # Under Slope
        if obj.type in (ObjType.FoldedSlope, ObjType.UnfoldedSlopeBody):
            if not is_move_heading_forward:
                return False

        backward_n_x, backward_n_y = next_step(x, y, (yaw + 2) % 4)
        is_move_heading_backward = backward_n_x == next_x and backward_n_y == next_y

        # Under Block
        if obj.type in (ObjType.Block, ObjType.Flag):
            if not (is_move_heading_forward or is_move_heading_backward):
                return False

        # ======= Transition Validation =======
        if not next_blow_obj.can_stand and next_blow_obj.type is not ObjType.UnfoldedSlopeBody:
            return False

        # Collision between smartcars
        if not ignore_cars:
            smartcars = self._blackboard.smartcars
            smartcars_pos = [
                (car.x, car.y, car.z)
                for i, car in enumerate(smartcars) if i != id_
            ]
            n_pos = (next_x, next_y, z)
            if any(n_pos == car_pos for car_pos in smartcars_pos):
                return False

        # Collision by lifted FoldedSlopeGear or lifted object
        if is_lift and moving_over_slope == 0:
            if not ignore_cars:
                gear_pose = [
                    (*next_step(car.x, car.y, car.lift_obj.yaw), car.z, car.lift_obj.yaw)
                    for i, car in enumerate(smartcars)
                    if car.is_lift and car.lift_obj.type is ObjType.FoldedSlope and i != id_
                ]
                # collision between lifted obj and other lifted gears
                if any(n_pos == g_pose[:3] for g_pose in gear_pose):
                    return False
            if lift_obj.type is ObjType.FoldedSlope:
                gear_x, gear_y = next_step(x, y, (yaw + 2) % 4)
                gear_n_x, gear_n_y = gear_x + move_dir[0], gear_y + move_dir[1]
                gear_n_pos_obj = grid[gear_n_x][gear_n_y][z]
                if gear_n_pos_obj.type is ObjType.FoldedSlopeGear:
                    # collision between lifted gear and gear on grid
                    if abs(gear_n_pos_obj.yaw - lift_obj.yaw) != 2:
                        return False
                # collision between lifted gear and other obj
                elif gear_n_pos_obj.type is not ObjType.Air:
                    return False
                if not ignore_cars:
                    lifted_obj_pos = [
                        (car.x, car.y, car.z)
                        for i, car in enumerate(smartcars) if i != id_ and car.is_lift
                    ]
                    gear_n_pos = (gear_n_x, gear_n_y, z)
                    # collision between lifted gear and other lifted obj (gear or block or slope)
                    if any(obj_pos == gear_n_pos for obj_pos in lifted_obj_pos) or \
                       any(g_pose[:3] == gear_n_pos and abs(g_pose[3] - lift_obj.yaw) != 2 for g_pose in gear_pose):
                        return False

        # Enter Air or Move down to Slope
        if next_obj.type is ObjType.Air:
            # move to free space
            if next_blow_obj.type is not ObjType.UnfoldedSlopeBody:
                return True
            # move down upon slope
            else:
                if obj.type not in (ObjType.Air, ObjType.Block, ObjType.Flag):
                    return False
                if not (yaw == next_blow_obj.yaw and is_move_heading_forward):
                    return False
                if next_blow_obj.obj_on_it != -1:            # one step
                    return False
                nn_x, nn_y = next_step(next_x, next_y, yaw)  # two steps
                if grid[nn_x][nn_y][z - 2].obj_on_it >= 0:
                    return False
                nnn_x, nnn_y = next_step(nn_x, nn_y, yaw)    # three steps
                bbelow_nnn_obj = grid[nnn_x][nnn_y][z - 2]
                return bbelow_nnn_obj.obj_on_it == -1 and bbelow_nnn_obj.can_stand

        # Enter Block
        if next_obj.type in (ObjType.Block, ObjType.Flag):
            if is_lift:
                return False
            return is_move_heading_forward or is_move_heading_backward

        # Enter FoldedSlope、UnfoldedSlopeBody
        if next_obj.type in (ObjType.FoldedSlope, ObjType.UnfoldedSlopeBody):
            if is_lift:
                return False
            valid_yaw = (next_obj.yaw + 2) % 4
            return valid_yaw == yaw and is_move_heading_backward

        # Enter FoldedSlopeGear
        if next_obj.type is ObjType.FoldedSlopeGear:
            return not is_lift

        # Enter UnfoldedSlopeFoot, move onto slope
        if next_obj.type is ObjType.UnfoldedSlopeFoot:
            if obj.type is not ObjType.Air:
                return False
            valid_yaw = (next_obj.yaw + 2) % 4
            if not (valid_yaw == yaw and is_move_heading_forward):
                return False
            if grid[next_x][next_y][z - 1].obj_on_it >= 0:  # one step
                return False
            nn_x, nn_y = next_step(next_x, next_y, yaw)     # two steps
            if grid[nn_x][nn_y][z].obj_on_it != -1:
                return False
            nnn_x, nnn_y = next_step(nn_x, nn_y, yaw)       # three steps
            nnn_obj = grid[nnn_x][nnn_y][z]
            above_nnn_obj = grid[nnn_x][nnn_y][z + 1]
            return nnn_obj.can_stand \
                and nnn_obj.obj_on_it < 0 \
                and (
                    above_nnn_obj.type is ObjType.Air
                    or (above_nnn_obj.type in (ObjType.Block, ObjType.Flag) and not is_lift)
                )

        return False

    def stop_action_mask(self, _, id_):
        [_, _, _, _, _, _, moving_over_slope] = self.get_smartcar_state(id_)
        return not moving_over_slope
