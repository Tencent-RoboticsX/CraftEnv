import itertools
from enum import IntEnum, auto, unique
from math import cos, sin

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from .grid_objs import ObjType


@unique
class Direction(IntEnum):
    """
    World coordinate system
    DIR_0:       -- >
    yaw = 0

    DIR_1:         ^
                   |
    yaw = np.pi * 0.5

    DIR_2:       < --
    yaw = np.pi * 1

    DIR_3:         |
                   v
    yaw = np.pi * 1.5
    """

    DIR_0 = 0
    DIR_1 = auto()
    DIR_2 = auto()
    DIR_3 = auto()


def next_step(x, y, theta):
    next_x = x + round(cos(theta))
    next_y = y + round(sin(theta))
    return next_x, next_y


def world_to_local(move_dir, yaw):
    theta = yaw * np.pi / 2
    r = R.from_matrix(
        [[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]]
    )
    move_dir = np.append(move_dir, 0)
    a = r.inv().apply(move_dir)
    return a[:2].astype(np.int)


def is_move_action(action):
    return 6 <= action <= 9


def is_smartcar_on_slope(smartcar):
    return smartcar.moving_over_slope


def save_scene2yaml(blackboard, step, timestamp, saved_path):
    for smartcar in blackboard.smartcars:
        if is_smartcar_on_slope(smartcar):
            return
    step = str(step).zfill(5)
    timestamp = f"{timestamp}-{step}"
    blackboard._bullet_client.addUserDebugText(text=timestamp,
                                               textPosition=[6, 12, 7],
                                               textColorRGB=[0, 0, 1],
                                               lifeTime=0.7,
                                               textSize=1.2,
                                               )
    grid_items = {
        "yaml_generated_time": timestamp,
        "area_length": blackboard.area_size[0],
        "area_width": blackboard.area_size[1],
        "area_height": blackboard.area_size[2],
        "wall_num": 0,
        "block_num": 0,
        "slope_num": 0,
        "smartcar_num": 0,
        "legged_robot_num": 2,
    }
    wall_list = []
    block_list = []
    fold_slope_list = []
    unfold_slope_list = []
    flag_list = []
    goal_list = []
    smartcar_list = []
    legged_robot_list = [
        {"id": 0, "x": 1, "y": 1, "z": 1, "yaw": 0},
        {"id": 1, "x": 1, "y": 2, "z": 1, "yaw": 0},
    ]

    grid = blackboard.grid
    wall_id = 0
    block_id = 0
    slope_id = 0
    for i, j, k in itertools.product(
        range(blackboard.area_size[0]),
        range(blackboard.area_size[1]),
        range(1, blackboard.area_size[2] + 1),
    ):
        obj = grid[i][j][k]

        if obj.type is ObjType.Wall:
            wall_list.append({"id": wall_id, "x": i, "y": j, "z": k})
            wall_id += 1

        if obj.type is ObjType.Block:
            block_list.append({"id": block_id, "x": i, "y": j, "z": k})
            block_id += 1

        if obj.type is ObjType.FoldedSlope:
            fold_slope_list.append(
                {"id": slope_id, "x": i, "y": j, "z": k, "yaw": obj.yaw}
            )
            slope_id += 1

        if obj.type is ObjType.UnfoldedSlopeBody:
            unfold_slope_list.append(
                {"id": slope_id, "x": i, "y": j, "z": k, "yaw": obj.yaw}
            )
            slope_id += 1

        if obj.type is ObjType.Flag:
            flag_list.append({"id": 0, "x": i, "y": j, "z": k})

    for smartcar_id, smartcar in enumerate(blackboard.smartcars):
        smartcar_list.append(
            {
                "id": int(smartcar_id),
                "x": int(smartcar.x),
                "y": int(smartcar.y),
                "z": int(smartcar.z),
                "yaw": int(smartcar.yaw),
            }
        )
        if smartcar.is_lift:
            if smartcar.lift_obj.type is ObjType.Block:
                block_list.append(
                    {"id": int(block_id), "x": int(smartcar.x), "y": int(smartcar.y), "z": int(smartcar.z)}
                )
                block_id += 1
            elif smartcar.lift_obj.type is ObjType.FoldedSlope:
                fold_slope_list.append(
                    {
                        "id": int(slope_id),
                        "x": int(smartcar.x),
                        "y": int(smartcar.y),
                        "z": int(smartcar.z),
                        "yaw": int(smartcar.lift_obj.yaw),
                    }
                )
                slope_id += 1
            elif smartcar.lift_obj.type is ObjType.Flag:
                flag_list.append(
                    {"id": 0, "x": int(smartcar.x), "y": int(smartcar.y), "z": int(smartcar.z)}
                )

    goal_list.append(
        {
            "id": 0,
            "x": int(blackboard.goal.x),
            "y": int(blackboard.goal.y),
            "z": int(blackboard.goal.z),
        }
    )

    grid_items.update({"wall_num": wall_id})
    grid_items.update({"block_num": block_id})
    grid_items.update({"slope_num": slope_id})
    grid_items.update({"smartcar_num": len(blackboard.smartcars)})
    with open(f"{saved_path}/{timestamp}.yaml", "w") as f:
        yaml.dump_all(
            [
                grid_items,
                {"block": block_list},
                {"fold_slope": fold_slope_list},
                {"unfold_slope": unfold_slope_list},
                {"smartcar": smartcar_list},
                {"flag": flag_list},
                {"goal": goal_list},
                {"legged_robot": legged_robot_list},
                {"wall": wall_list},
            ],
            f,
            sort_keys=False,
        )


def save_action2yaml(record_action, record_action_order, timestamp, saved_path):
    action = np.array(record_action)
    action_order = np.array(record_action_order)
    data = np.concatenate((action, action_order), axis=1)
    # saved format: action, action order
    with open(f"{saved_path}/{timestamp}_record_action.yaml", "w") as f:
        np.savetxt(f, data, fmt='%i')


def load_action2yaml(robot_num, saved_path):
    with open(f"{saved_path}", "r") as f:
        data = np.loadtxt(f, dtype=int)
        action, action_order = data[:, :robot_num], data[:, robot_num:]
        return action, action_order
