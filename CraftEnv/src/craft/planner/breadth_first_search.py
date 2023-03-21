from collections import deque
import numpy as np
from .smartcar_planner import SmartCarPlanner
from ..grid_objs import ObjType


class BreadthFirstSearch(SmartCarPlanner):
    def __init__(self, blackboard):
        super().__init__(blackboard)

    def is_inbound(self, x, y, z):
        if x < self.x_min or x > self.x_max:
            return False
        if y < self.y_min or y > self.y_max:
            return False
        if z < self.z_min or z > self.z_max:
            return False
        return True

    def set_bound(self, size, start_x, start_y):
        if size is None:
            self.x_min, self.x_max = 0, self.length - 1
            self.y_min, self.y_max = 0, self.width - 1
            self.z_min, self.z_max = 0, self.height - 1
            return

        if isinstance(size, int):
            l, w, h = size, size, size
        elif isinstance(size, (tuple, list)):
            l, w, h = size
        else:
            raise TypeError

        self.x_min = max(0, start_x - l / 2)
        self.x_max = min(self.length - 1, start_x + l / 2)
        self.y_min = max(0, start_y - w / 2)
        self.y_max = min(self.width - 1, start_y + w / 2)
        self.z_min = 0
        self.z_max = min(self.height - 1, h)

    def can_move(self, node, move_dir):
        x, y, z, yaw, moving_over_slope = node
        kw = dict(
            x=x,
            y=y,
            z=z,
            yaw=yaw,
            is_lift=False,
            lift_obj=None,
            moving_over_slope=moving_over_slope
        )
        return self.action_mask_proxy.move_action_mask(move_dir, self.agent_id, kw, ignore_cars=self.ignore_cars)

    def can_rotate(self, node, rotate_dir):
        x, y, z, yaw, moving_over_slope = node
        kw = dict(
            x=x,
            y=y,
            z=z,
            yaw=yaw,
            is_lift=False,
            lift_obj=None,
            moving_over_slope=moving_over_slope
        )
        return self.action_mask_proxy.rotate_mask(rotate_dir, self.agent_id, kw, ignore_cars=self.ignore_cars)

    def get_moved_node(self, curr_node, move_dir):
        x, y, z, yaw, moving_over_slope = curr_node
        x += move_dir[0]
        y += move_dir[1]

        obj = self.grid[x][y][z]
        blow_obj = self.grid[x][y][z - 1]
        if obj.type is ObjType.UnfoldedSlopeFoot and moving_over_slope == 0:
            moving_over_slope = 1
        elif obj.type is ObjType.UnfoldedSlopeBody and moving_over_slope == 1:
            moving_over_slope = 2
            z += 1
        elif blow_obj.type is ObjType.UnfoldedSlopeBody and moving_over_slope == 0:
            moving_over_slope = 2
        elif blow_obj.type is ObjType.UnfoldedSlopeFoot and moving_over_slope == 2:
            moving_over_slope = 1
            z -= 1
        else:
            moving_over_slope = 0
        return (x, y, z, yaw, moving_over_slope)

    def get_rotated_node(self, curr_node, rotate_dir):
        x, y, z, yaw, moving_over_slope = curr_node
        yaw = (yaw + rotate_dir) % 4
        return (x, y, z, yaw, moving_over_slope)

    def get_successors(self, curr_node):
        successors = []

        # move action
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if not self.can_move(curr_node, (dx, dy)):
                continue
            node = self.get_moved_node(curr_node, (dx, dy))
            if not self.is_inbound(*node[:3]):
                continue
            successors.append(node)

        # rotate action
        for d_yaw in [-1, 1]:
            if not self.can_rotate(curr_node, d_yaw):
                continue
            node = self.get_rotated_node(curr_node, d_yaw)
            successors.append(node)

        return successors

    def search(self, agent_id, start_x, start_y, start_z, yaw,
               moving_over_slope=0, visualize=False, view_size=None, ignore_cars=False):
        """
        :param size: int or tuple, specify bounded search space. If None, bfs searches the entire space.
        """
        self.agent_id = agent_id
        self.reset()
        self.set_bound(view_size, start_x, start_y)
        self.ignore_cars = ignore_cars

        visited = np.zeros((self.length, self.width, self.height))
        close_set = np.zeros((self.length, self.width, self.height, 4))
        open_set = np.zeros((self.length, self.width, self.height, 4))

        q = deque()
        start_node = (start_x, start_y, start_z, yaw, moving_over_slope)
        q.append(start_node)
        open_set[start_node[:4]] = 1

        block_length = self.blackboard.BLOCK_LENGTH
        block_height = self.blackboard.BLOCK_HEIGHT
        if visualize:
            import pybullet_data
            self.blackboard._bullet_client.setAdditionalSearchPath(
                pybullet_data.getDataPath())
            vis_obj_list = []

            self.blackboard._bullet_client.configureDebugVisualizer(
                self.blackboard._bullet_client.COV_ENABLE_RENDERING, 0)

        while q:
            curr_node = q.popleft()
            open_set[curr_node[:4]] = 0
            close_set[curr_node[:4]] = 1
            visited[curr_node[:3]] = 1

            if visualize:
                x, y, z = curr_node[:3]
                vis_x = x * block_length + block_length / 2
                vis_y = y * block_length + block_length / 2
                vis_z = (z - 1) * block_height + block_height / 2
                handle = self.blackboard._bullet_client.loadURDF(
                    'cube.urdf', (vis_x, vis_y, vis_z), [1, 0, 0, 1], globalScaling=0.05)
                for iii in range(self.blackboard._bullet_client.getNumJoints(handle)):
                    self.blackboard._bullet_client.changeVisualShape(
                        handle, iii, rgbaColor=[1, 0, 0, 1]
                    )
                vis_obj_list.append(handle)

            successed_nodes = self.get_successors(curr_node)
            for node in successed_nodes:
                if close_set[node[:4]] or open_set[node[:4]]:
                    continue

                q.append(node)
                open_set[node[:4]] = 1

        if visualize:
            self.blackboard._bullet_client.configureDebugVisualizer(
                self.blackboard._bullet_client.COV_ENABLE_RENDERING, 1)
            input('===== Enter to remove BFS blocks =====\n')
            self.blackboard._bullet_client.configureDebugVisualizer(
                self.blackboard._bullet_client.COV_ENABLE_RENDERING, 0)
            for i in vis_obj_list:
                self.blackboard._bullet_client.removeBody(i)
            self.blackboard._bullet_client.configureDebugVisualizer(
                self.blackboard._bullet_client.COV_ENABLE_RENDERING, 1)
        return visited
