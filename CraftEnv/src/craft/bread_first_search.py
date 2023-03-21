from collections import deque
import numpy as np
from craftenv.sim_envs.pybullet_envs.craft.grid_objs import (
    Air, Block, Flag, FoldedSlope, FoldedSlopeGear, UnfoldedSlopeBody, UnfoldedSlopeFoot
)
from craftenv.sim_envs.pybullet_envs.craft.utils import next_step


class BreadthFirstSearch:

    def __init__(self, blackboard):
        """
        Initialize grid map for bfs search
        """
        self._blackboard = blackboard
        self.motion = self.get_motion_model()
        self.area_size = blackboard.area_size

    def calc_reachable_space(self, x, y, z):
        grid = self._blackboard.grid
        maxx = self._blackboard.area_size[0] + 1
        minx = 0
        maxy = self._blackboard.area_size[1] + 1
        miny = 0
        maxz = self._blackboard.area_size[2]
        minz = 1
        visited = np.zeros((maxx, maxy, maxz + 1))

        q = deque()
        q.append((x, y, z))
        visited[x][y][z] = 1
        cnt = 1

        while q:
            x, y, z = q.popleft()

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                n_x, n_y, n_z = x + self.motion[i][0], y + self.motion[i][1], z

                if minx <= n_x < maxx and \
                   miny <= n_y < maxy and \
                   minz <= n_z <= maxz:
                    n_obj = grid[n_x][n_y][n_z]
                    blow_n_obj = grid[n_x][n_y][n_z - 1]

                    if visited[n_x][n_y][n_z] == 0:
                        if blow_n_obj.can_stand and isinstance(n_obj, (Air, Block, Flag, FoldedSlopeGear)):
                            q.append((n_x, n_y, n_z))
                            visited[n_x][n_y][n_z] = 1
                            cnt += 1

                        elif isinstance(n_obj, UnfoldedSlopeFoot):
                            yaw = (n_obj.yaw + 2) % 4
                            nn_x, nn_y = next_step(n_x, n_y, yaw * np.pi / 2)
                            q.append((n_x, n_y, n_z))
                            visited[n_x][n_y][n_z] = 1
                            cnt += 1
                            if minx <= nn_x < maxx and \
                               miny <= nn_y < maxy and \
                               minz <= n_z + 1 <= maxz:
                                q.append((nn_x, nn_y, n_z + 1))
                                visited[nn_x][nn_y][n_z + 1] = 1
                                cnt += 1

                        elif isinstance(n_obj, (FoldedSlope, UnfoldedSlopeBody)):
                            pre_yaw = (n_obj.yaw + 2) % 4
                            pre_x, pre_y = next_step(n_x, n_y, pre_yaw * np.pi / 2)
                            if pre_x == x and pre_y == y:
                                visited[n_x][n_y][n_z] = 1
                                cnt += 1

        return visited

    @staticmethod
    def get_motion_model():
        return np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
