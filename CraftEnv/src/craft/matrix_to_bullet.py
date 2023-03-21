import itertools
import numpy as np
from scipy.spatial.transform import Rotation as R
from .bullet_block import BulletBlocks
from .bullet_wall import BulletWalls
from .bullet_flag import BulletFlag
from .bullet_goal import BulletGoal
from .bullet_slope import BulletSlope
from .bullet_smartcar import BulletSmartcar
from .grid_objs import ObjType


class MatrixToBullet:
    def __init__(self, bullet_client, blackboard):
        self._bullet_client = bullet_client
        self._blackboard = blackboard
        self.area_size = blackboard.area_size
        self.block_length = blackboard.BLOCK_LENGTH
        self.block_height = blackboard.BLOCK_HEIGHT

        bullet_client.resetDebugVisualizerCamera(
            cameraDistance=2,
            cameraYaw=0,
            cameraPitch=-30,
            cameraTargetPosition=[5, 0, 4],
        )

        bullet_client.setPhysicsEngineParameter(collisionFilterMode=0)
        bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_RENDERING, 0)

        self.blocks = BulletBlocks(bullet_client, blackboard)
        self.slopes = [
            BulletSlope(bullet_client) for _ in range(self._blackboard.slope_num)
        ]
        self.smartcars = [
            BulletSmartcar(bullet_client) for _ in range(self._blackboard.smartcar_num)
        ]
        self.flag = BulletFlag(bullet_client)
        self.goal = BulletGoal(bullet_client)
        if self._blackboard.wall_num != 0:
            self.walls = BulletWalls(bullet_client, blackboard)

        bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_RENDERING, 1)

        def yaw_to_quaternion(yaw):
            r = R.from_euler("z", yaw, degrees=False)
            return r.as_quat()

        self.yaw_to_quat = {
            0: yaw_to_quaternion(0 * np.pi / 2),
            1: yaw_to_quaternion(1 * np.pi / 2),
            2: yaw_to_quaternion(2 * np.pi / 2),
            3: yaw_to_quaternion(3 * np.pi / 2),
        }

    def get_bullet_position(self, i, j, k):
        x = i * self.block_length + 0.5 * self.block_length
        y = j * self.block_length + 0.5 * self.block_length
        z = k * self.block_height
        return [x, y, z]

    def sync(self):
        grid = self._blackboard.grid
        wall_id = 0
        block_id = 0
        slope_id = 0
        for i, j, k in itertools.product(
            range(self.area_size[0]),
            range(self.area_size[1]),
            range(1, self.area_size[2] + 1),
        ):
            obj = grid[i][j][k]

            if self._blackboard.wall_num != 0 and obj.type is ObjType.Wall:
                position = self.get_bullet_position(i, j, k)
                self._bullet_client.resetBasePositionAndOrientation(
                    self.walls.ids[wall_id], position, self.yaw_to_quat[0]
                )
                wall_id += 1

            if obj.type is ObjType.Block:
                position = self.get_bullet_position(i, j, k)
                self._bullet_client.resetBasePositionAndOrientation(
                    self.blocks.ids[block_id], position, self.yaw_to_quat[0]
                )
                block_id += 1

            if obj.type is ObjType.FoldedSlope:
                position = self.get_bullet_position(i, j, k - 1)
                self._bullet_client.resetBasePositionAndOrientation(
                    self.slopes[slope_id].robot_id, position, self.yaw_to_quat[obj.yaw]
                )
                self.slopes[slope_id].fold()
                slope_id += 1

            if obj.type is ObjType.UnfoldedSlopeBody:
                position = self.get_bullet_position(i, j, k - 1)
                self._bullet_client.resetBasePositionAndOrientation(
                    self.slopes[slope_id].robot_id, position, self.yaw_to_quat[obj.yaw]
                )
                self.slopes[slope_id].unfold()
                slope_id += 1

            if obj.type is ObjType.Flag:
                position = self.get_bullet_position(i, j, k - 1)
                self._bullet_client.resetBasePositionAndOrientation(
                    self.flag.robot_id, position, self.yaw_to_quat[0]
                )

        for smartcar_id, smartcar in enumerate(self._blackboard.smartcars):
            position = self.get_bullet_position(smartcar.x, smartcar.y, smartcar.z - 1)
            position[2] += 0.043
            self._bullet_client.resetBasePositionAndOrientation(
                self.smartcars[smartcar_id].robot_id,
                position,
                self.yaw_to_quat[smartcar.yaw],
            )
            if smartcar.is_lift:
                if smartcar.lift_obj.type is ObjType.Block:
                    position = self.get_bullet_position(
                        smartcar.x, smartcar.y, smartcar.z + 1
                    )
                    self._bullet_client.resetBasePositionAndOrientation(
                        self.blocks.ids[block_id], position, self.yaw_to_quat[0]
                    )
                    block_id += 1
                elif smartcar.lift_obj.type is ObjType.FoldedSlope:
                    position = self.get_bullet_position(
                        smartcar.x, smartcar.y, smartcar.z
                    )
                    self._bullet_client.resetBasePositionAndOrientation(
                        self.slopes[slope_id].robot_id,
                        position,
                        self.yaw_to_quat[smartcar.lift_obj.yaw],
                    )
                    self.slopes[slope_id].fold()
                    slope_id += 1
                elif smartcar.lift_obj.type is ObjType.Flag:
                    position = self.get_bullet_position(
                        smartcar.x, smartcar.y, smartcar.z
                    )
                    self._bullet_client.resetBasePositionAndOrientation(
                        self.flag.robot_id, position, self.yaw_to_quat[0]
                    )

        goal_coord = self.get_bullet_position(
            self._blackboard.goal.x,
            self._blackboard.goal.y,
            self._blackboard.goal.z - 1,
        )
        self._bullet_client.resetBasePositionAndOrientation(
            self.goal.robot_id, goal_coord, self.yaw_to_quat[0]
        )

        self._bullet_client.stepSimulation()
