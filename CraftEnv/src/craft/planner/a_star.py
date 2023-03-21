"""
Implementation of A* planning algorithm for 4-connected grid maps.
"""
import math
from heapq import heappop, heappush, heapify
from ..action_enum import ActionEnum
from .smartcar_planner import SmartCarNode, SmartCarPlanner


class AStarNode(SmartCarNode):
    def __init__(self,
                 x,
                 y,
                 z=1,
                 yaw=0,
                 is_lift=False,
                 lift_obj=None,
                 moving_over_slope=0,
                 parent=None,
                 heuristic_cost=0.,
                 dijkstra_cost=0.,
                 depth=0):
        super().__init__(x, y, z, yaw, is_lift, lift_obj, moving_over_slope)
        self.d_cost = dijkstra_cost
        self.cost = heuristic_cost + dijkstra_cost
        self.parent = parent
        self.depth = depth

    def __lt__(self, other) -> bool:
        return self.cost < other.cost

    def copy(self):
        return AStarNode(
            x=self.x,
            y=self.y,
            z=self.z,
            yaw=self.yaw,
            is_lift=self.is_lift,
            lift_obj=self.new_obj(self.lift_obj),
            moving_over_slope=self.moving_over_slope,
            parent=None,
            heuristic_cost=0.,
            dijkstra_cost=0.,
            depth=self.depth)


class OpenList:
    def __init__(self) -> None:
        self._heap = []
        self._pos2node = {}

    def push(self, node):
        heappush(self._heap, node)
        self._pos2node[node.key] = node

    def pop(self):
        node = heappop(self._heap)
        self._pos2node.pop(node.key)
        return node

    def update(self, node):
        item = self._pos2node[node.key]
        if item.d_cost > node.d_cost:
            # item.t = node.t
            item.cost -= item.d_cost - node.d_cost
            item.d_cost = node.d_cost
            item.parent = node.parent
            heapify(self._heap)

    def exist(self, node):
        return False if node is None else node.key in self._pos2node

    def reset(self):
        self._heap.clear()
        self._pos2node.clear()

    def empty(self):
        return len(self._heap) == 0


class CloseList:
    def __init__(self) -> None:
        self._set = set()

    def exist(self, node):
        return False if node is None else node.key in self._set

    def reset(self):
        self._set.clear()

    def add(self, node):
        self._set.add(node.key)

    def empty(self):
        return len(self._set) == 0


class AStarPlanner(SmartCarPlanner):
    Node = AStarNode

    def __init__(self, blackboard, search_depth=10) -> None:
        super().__init__(blackboard)
        self.open_list = OpenList()
        self.close_list = CloseList()

        self.start_node = None
        self.goal_node = None

        self.search_depth = search_depth

    @staticmethod
    def euclidean_distance(n1, n2):
        return math.hypot(n1.x - n2.x, n1.y - n2.y)

    @staticmethod
    def manhattan_distance(n1, n2):
        return abs(n1.x - n2.x) + abs(n1.y - n2.y)

    def calc_heuristic_cost(self, n1, n2, weight=1.0):
        """
        use manhattan distance for *orthogonal* grid map,
        which is the optimal heuristic function.
        """
        return weight * self.manhattan_distance(n1, n2)

    def get_final_path(self):
        path = []
        node = self.goal_node
        while node is not None:
            path.append((node.x, node.y, node.z, node.yaw))
            node = node.parent
        path.reverse()
        return path

    def get_final_cost(self):
        return self.goal_node.cost

    @staticmethod
    def pos_to_action(p1, p2):
        x1, y1, _, yaw1 = p1
        x2, y2, _, yaw2 = p2

        if (yaw1 + 1) % 4 == yaw2:
            return ActionEnum.ROTATE_LEFT
        if (yaw1 - 1) % 4 == yaw2:
            return ActionEnum.ROTATE_RIGHT

        dx, dy = x2 - x1, y2 - y1
        if dy == 0:
            if dx == 1:
                return ActionEnum.MOVE_RIGHT
            if dx == -1:
                return ActionEnum.MOVE_LEFT
        if dx == 0:
            if dy == 1:
                return ActionEnum.MOVE_FORWARD
            if dy == -1:
                return ActionEnum.MOVE_BACK

        return ActionEnum.STOP

    def get_actions(self, path):
        if not path or len(path) < 2:
            return [ActionEnum.STOP]
        return [self.pos_to_action(path[t], path[t + 1]) for t in range(len(path) - 1)]

    def get_successors(self, curr_node: AStarNode):
        successors = []

        # move action
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if not self.can_move(curr_node, (dx, dy)):
                continue
            node = self.get_moved_node(curr_node, (dx, dy))
            node.depth = curr_node.depth + 1
            node.d_cost = curr_node.d_cost + self.move_cost
            h_cost = self.calc_heuristic_cost(
                node, self.goal_node, weight=self.move_cost)
            node.cost = node.d_cost + h_cost
            node.parent = curr_node
            successors.append(node)

        # rotate action
        for d_yaw in [-1, 1]:
            if not self.can_rotate(curr_node, d_yaw):
                continue
            node = self.get_rotated_node(curr_node, d_yaw)
            node.depth = curr_node.depth + 1
            node.d_cost = curr_node.d_cost + self.rotate_cost
            h_cost = self.calc_heuristic_cost(
                node, self.goal_node, weight=self.move_cost)
            node.cost = node.d_cost + h_cost
            node.parent = curr_node
            successors.append(node)

        return successors

    def _plan(self, start_node, goal_node, verbose):
        assert self.grid is not None, \
            'Grid map not specified, please call set_grid() before planning'

        self.start_node = start_node
        self.goal_node = goal_node

        self.open_list.reset()
        self.close_list.reset()

        self.open_list.push(self.start_node)

        while not self.open_list.empty():
            curr_node = self.open_list.pop()

            if curr_node.pos == self.goal_node.pos:
                self.goal_node = curr_node
                path = self.get_final_path()
                if verbose:
                    print('Planning succeeded.')
                    print(f'Cost: {self.goal_node.cost}')
                    print(path)
                return path

            # early stopping for max_depth truncation
            if curr_node.depth > self.search_depth:
                break

            self.close_list.add(curr_node)  # add to visited set

            successed_nodes = self.get_successors(curr_node)

            for node in successed_nodes:
                if self.close_list.exist(node):  # already visited
                    continue

                if not self.open_list.exist(node):  # maintain open set
                    self.open_list.push(node)
                else:
                    self.open_list.update(node)

        if verbose:
            print('Planning failed. No feasible path.')
        return []
