import itertools

import numpy as np
import yaml

from .grid_objs import Air, Ground, ObjType, Wall
from .utils import Direction


class Point:
    def __init__(self, x, y, z=1):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Point(self.x + other, self.y + other)

    def __mul__(self, other):
        return Point(self.x * other, self.y * other)

    def __hash__(self):
        return 1

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __repr__(self):
        return f"Point({self.x}, {self.y}, {self.z})"


class Blackboard:
    BLOCK_LENGTH = 0.3225
    BLOCK_HEIGHT = 0.155 + 0.03
    SMARTCAR_LENGTH = 0.155
    SMARTCAR_WIDTH = 0.155
    SMARTCAR_HEIGHT = 0.155

    def __init__(self, blueprint_path_list):
        """
        ^ y
        |
        |
        |
        o ----------> x
        world coordinate
        """
        self.spawn_point_set = set()
        self.blueprint_path_list = blueprint_path_list
        self.area_size = [0, 0, 0]  # length, width, height
        self.load_blueprint(blueprint_path_list[0])

    def load_blueprint(self, blueprint_path):
        f = open(blueprint_path, "r", encoding="utf-8")
        self.blueprint_path = blueprint_path
        template_generator = yaml.safe_load_all(f)
        self.template = {}
        for t in template_generator:
            if t is not None:
                self.template.update(t)

        try:
            self.wall_num = self.template["wall_num"]
        except KeyError as e:
            self.wall_num = 0
            print("KeyError, ", e)
        self.block_num = self.template["block_num"]
        self.slope_num = self.template["slope_num"]
        self.smartcar_num = self.template["smartcar_num"]
        self.legged_robot_num = self.template["legged_robot_num"]
        self.area_size[0] = self.template["area_length"]
        self.area_size[1] = self.template["area_width"]
        self.area_size[2] = self.template["area_height"]

    def reset(self, blueprint_path=None):
        if blueprint_path is not None:
            self.load_blueprint(blueprint_path)

        self.spawn_point_set.clear()

        length = self.area_size[0] + 1
        width = self.area_size[1] + 1
        height = self.area_size[2] + 1

        self.grid = [
            [[Air() for _ in range(height)] for _ in range(width)]
            for _ in range(length)
        ]

        for i, j in itertools.product(range(length), range(width)):
            self.grid[i][j][0] = Ground()
        for k in range(height):
            for i in range(width):
                self.grid[-1][i][k] = Wall()
            for j in range(length):
                self.grid[j][-1][k] = Wall()

    def random_spawn_obj(self, obj_type):
        direction = np.random.randint(Direction.DIR_0, Direction.DIR_3)
        random_cnt = 0
        while True:
            random_cnt += 1
            assert random_cnt < 1000, f"please reduce obj {obj_type} num"
            p = Point(
                np.random.randint(0, self.area_size[0]),
                np.random.randint(0, self.area_size[1]),
            )
            obj = self.grid[p.x][p.y][p.z]
            blow_obj = self.grid[p.x][p.y][p.z - 1]
            if (
                p not in self.spawn_point_set
                and obj.type is ObjType.Air
                and blow_obj.can_stand
            ):
                break
        return p, direction
