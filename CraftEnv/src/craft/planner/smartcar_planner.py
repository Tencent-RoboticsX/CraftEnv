"""
A meta planner for smartcars planning in multi-flat 4-connected grid maps,
which supports planning with rotation and non-diagonal movement under action_mask constraints.
"""
from ..grid_objs import FoldedSlope
from ..grid_objs import ObjType


class SmartCarNode:
    def __init__(self,
                 x,
                 y,
                 z=1,
                 yaw=0,
                 is_lift=False,
                 lift_obj=None,
                 moving_over_slope=0):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.is_lift = is_lift
        self.lift_obj = lift_obj
        self.moving_over_slope = moving_over_slope

    @property
    def key(self):
        return (self.x, self.y, self.z, self.yaw)

    @property
    def pos(self):
        return (self.x, self.y, self.z)

    def copy(self):
        return SmartCarNode(
            x=self.x,
            y=self.y,
            z=self.z,
            yaw=self.yaw,
            is_lift=self.is_lift,
            lift_obj=self.new_obj(self.lift_obj),
            moving_over_slope=self.moving_over_slope)

    def new_obj(self, obj):
        return FoldedSlope(obj.yaw) if obj is not None and obj.type is ObjType.FoldedSlope else obj


class SmartCarPlanner:
    Node = SmartCarNode

    def __init__(self, blackboard) -> None:
        self.move_cost = 1.0
        self.rotate_cost = 1.0
        self.blackboard = blackboard
        self.action_mask_proxy = None
        self.grid = None
        self.agent_id = None

    def reset(self):
        """
        Call blackboard.reset() before this function
        """
        self.action_mask_proxy = self.blackboard.action_mask_proxy
        self._set_grid(self.blackboard.grid)

    def _set_grid(self, grid):
        self.grid = grid
        self.length = len(grid)
        self.width = len(grid[0])
        self.height = len(grid[0][0])

    def is_inbound(self, x, y):
        if x < 0 or x >= self.length:
            return False
        if y < 0 or y >= self.width:
            return False
        return True

    def can_move(self, node: SmartCarNode, move_dir):
        if not self.is_inbound(node.x + move_dir[0], node.y + move_dir[1]):
            return False

        kw = dict(
            x=node.x,
            y=node.y,
            z=node.z,
            yaw=node.yaw,
            is_lift=node.is_lift,
            lift_obj=node.lift_obj,
            moving_over_slope=node.moving_over_slope
        )
        return self.action_mask_proxy.move_action_mask(move_dir, self.agent_id, kw)

    def can_rotate(self, node: SmartCarNode, rotate_dir):
        kw = dict(
            x=node.x,
            y=node.y,
            z=node.z,
            yaw=node.yaw,
            is_lift=node.is_lift,
            lift_obj=node.lift_obj,
            moving_over_slope=node.moving_over_slope
        )
        return self.action_mask_proxy.rotate_mask(rotate_dir, self.agent_id, kw)

    def get_moved_node(self, curr_node: SmartCarNode, move_dir):
        node = curr_node.copy()
        node.x += move_dir[0]
        node.y += move_dir[1]

        obj = self.grid[node.x][node.y][node.z]
        blow_obj = self.grid[node.x][node.y][node.z - 1]
        if obj.type is ObjType.UnfoldedSlopeFoot and node.moving_over_slope == 0:
            node.moving_over_slope = 1
        elif obj.type is ObjType.UnfoldedSlopeBody and node.moving_over_slope == 1:
            node.moving_over_slope = 2
            node.z += 1
        elif blow_obj.type is ObjType.UnfoldedSlopeBody and node.moving_over_slope == 0:
            node.moving_over_slope = 2
        elif blow_obj.type is ObjType.UnfoldedSlopeFoot and node.moving_over_slope == 2:
            node.moving_over_slope = 1
            node.z -= 1
        else:
            node.moving_over_slope = 0
        return node

    def get_rotated_node(self, curr_node: SmartCarNode, rotate_dir):
        node = curr_node.copy()
        node.yaw = (node.yaw + rotate_dir) % 4
        if node.is_lift and node.lift_obj.type is ObjType.FoldedSlope:
            node.lift_obj.yaw = (node.lift_obj.yaw + rotate_dir) % 4
        return node

    def get_successors(self, curr_node: SmartCarNode):
        successors = []

        # move action
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if not self.can_move(curr_node, (dx, dy)):
                continue
            node = self.get_moved_node(curr_node, (dx, dy))
            successors.append(node)

        # rotate action
        for d_yaw in [-1, 1]:
            if not self.can_rotate(curr_node, d_yaw):
                continue
            node = self.get_rotated_node(curr_node, d_yaw)
            successors.append(node)

        return successors

    def plan(self, agent_id, start_x, start_y, start_z, yaw, is_lift, lift_obj, moving_over_slope,
             goal_x, goal_y, goal_z, verbose=False):

        assert self.grid is not None, \
            'Grid map not specified, please call set_grid() before planning'

        self.agent_id = agent_id

        start_node = self.Node(
            x=start_x, y=start_y, z=start_z,
            yaw=yaw, is_lift=is_lift, lift_obj=lift_obj,
            moving_over_slope=moving_over_slope)

        goal_node = self.Node(x=goal_x, y=goal_y, z=goal_z)

        return self._plan(start_node, goal_node, verbose)

    def _plan(self, start_node, goal_node, verbose):
        raise NotImplementedError
