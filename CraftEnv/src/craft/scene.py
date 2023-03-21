import numpy as np

from craft import utils

from .blackboard import Point
from .grid_objs import (Block, Flag, FoldedSlope, FoldedSlopeGear, ObjType,
                        UnfoldedSlopeBody, UnfoldedSlopeFoot, Wall)


class Scene:

    def __init__(self, blackboard):
        self._blackboard = blackboard
        self.area_size = self._blackboard.area_size

    def checke_coord_legal(self, obj):
        if obj["x"] >= self.area_size[0] or \
           obj["y"] >= self.area_size[1] or \
           obj["z"] > self.area_size[2]:
            raise ValueError("object coordinate out of bounds")

    def random_place_block(self, max_id):
        i = 0
        while i < max_id:
            name = "block"
            p, _ = self._blackboard.random_spawn_obj(name)
            self._blackboard.grid[p.x][p.y][p.z] = Block()
            self._blackboard.spawn_point_set.add(p)
            i += 1

    def random_place_slope(self, max_id):
        i = 0
        while i < max_id:
            name = "slope"
            while True:
                p, direction = self._blackboard.random_spawn_obj(name)
                # +2 to consider the Gear
                pre_x, pre_y = utils.next_step(p.x, p.y,
                                               (direction + 2) * np.pi / 2)
                next_x, next_y = utils.next_step(p.x, p.y,
                                                 direction * np.pi / 2)
                if self._blackboard.grid[pre_x][pre_y][p.z].type is not \
                        ObjType.Wall and \
                        self._blackboard.grid[next_x][next_y][p.z].type \
                        is ObjType.Air:
                    break
            self._blackboard.grid[p.x][p.y][p.z] = FoldedSlope(direction)
            self._blackboard.grid[next_x][next_y][p.z] = FoldedSlopeGear(
                direction)
            self._blackboard.spawn_point_set.add(p)
            self._blackboard.spawn_point_set.add(Point(next_x, next_y, p.z))
            i += 1

    def z_axis(self, elem):
        return elem["z"]

    def reset(self):
        self.template = self._blackboard.template

        # must process block first!
        try:
            self.template["block"].sort(key=self.z_axis)
            for i in self.template["block"]:
                self.checke_coord_legal(i)
                self._blackboard.grid[i["x"]][i["y"]][i["z"]] = Block()
                self._blackboard.grid[i["x"]][i["y"]][i["z"] -
                                                      1].obj_on_it = -2
                self._blackboard.spawn_point_set.add(
                    Point(i["x"], i["y"], i["z"]))
                # place_block = i["id"] + 1
        except KeyError as e:
            print("KeyError, ", e)
            pass
        try:
            self.template["fold_slope"].sort(key=self.z_axis)
            for i in self.template["fold_slope"]:
                self.checke_coord_legal(i)
                self._blackboard.grid[i["x"]][i["y"]][i["z"]] = FoldedSlope(
                    i["yaw"])
                self._blackboard.grid[i["x"]][i["y"]][i["z"] -
                                                      1].obj_on_it = -2
                n_x, n_y = utils.next_step(i["x"], i["y"],
                                           i["yaw"] * np.pi / 2)
                self._blackboard.grid[n_x][n_y][i["z"]] = FoldedSlopeGear(
                    i["yaw"])
                self._blackboard.spawn_point_set.add(
                    Point(i["x"], i["y"], i["z"]))
                # place_slope = i["id"] + 1
        except KeyError as e:
            print("KeyError, ", e)
            pass
        try:
            self.template["unfold_slope"].sort(key=self.z_axis)
            for i in self.template["unfold_slope"]:
                self.checke_coord_legal(i)
                self._blackboard.grid[i["x"]][i["y"]][
                    i["z"]] = UnfoldedSlopeBody(i["yaw"])
                self._blackboard.grid[i["x"]][i["y"]][i["z"] -
                                                      1].obj_on_it = -2
                pre_x, pre_y = utils.next_step(i["x"], i["y"],
                                               (i["yaw"] + 2) * np.pi / 2)
                pre_obj = self._blackboard.grid[pre_x][pre_y][i["z"]]
                if isinstance(pre_obj, Block):
                    pre_obj.near_unfold_slope_body = True
                self._blackboard.spawn_point_set.add(
                    Point(i["x"], i["y"], i["z"]))
                n_x, n_y = utils.next_step(i["x"], i["y"],
                                           i["yaw"] * np.pi / 2)
                self._blackboard.grid[n_x][n_y][i["z"]] = UnfoldedSlopeFoot(
                    i["yaw"])
                self._blackboard.grid[n_x][n_y][i["z"] - 1].obj_on_it = -2
                front_x, front_y = utils.next_step(n_x, n_y,
                                                   i["yaw"] * np.pi / 2)
                front_blow_obj = self._blackboard.grid[front_x][front_y][i["z"]
                                                                         - 1]
                if isinstance(front_blow_obj, Block):
                    front_blow_obj.near_blow_unfold_slope_foot = True
                self._blackboard.spawn_point_set.add(Point(n_x, n_y, i["z"]))
                # place_slope = i["id"] + 1
        except KeyError as e:
            print("KeyError, ", e)
            pass
        try:
            for i in self.template["flag"]:
                self.checke_coord_legal(i)
                self._blackboard.grid[i["x"]][i["y"]][i["z"]] = Flag()
                self._blackboard.grid[i["x"]][i["y"]][i["z"] -
                                                      1].obj_on_it = -2
                self._blackboard.spawn_point_set.add(
                    Point(i["x"], i["y"], i["z"]))
        except KeyError as e:
            print("KeyError, ", e)
            pass
        try:
            for i in self.template["wall"]:
                self.checke_coord_legal(i)
                self._blackboard.grid[i["x"]][i["y"]][i["z"]] = Wall()
        except KeyError as e:
            print("KeyError, ", e)
            pass
        place_block = len(
            self.template["block"]) if "block" in self.template else 0
        place_slope = 0
        if "fold_slope" in self.template:
            place_slope = len(self.template["fold_slope"])
        if "unfold_slope" in self.template:
            place_slope += len(self.template["unfold_slope"])

        self.random_place_slope(self._blackboard.slope_num - place_slope)
        self.random_place_block(self._blackboard.block_num - place_block)
