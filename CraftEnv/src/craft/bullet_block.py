import os
from craft import get_urdf_path


class BulletBlock:
    def __init__(self, bullet_client):
        self._bullet_client = bullet_client
        self.init_pose = [0.0, 0.0, 1.0]
        self.init_quat = [0.0, 0.0, 0.0, 1.0]
        self._init_model(self.init_pose, self.init_quat)

    def _init_model(self, init_pose, init_quat):
        robot_path = os.path.join(get_urdf_path(), "block/block.urdf")
        self.robot_id = self._bullet_client.loadURDF(
            robot_path,
            init_pose,
            init_quat,
            flags=(self._bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )


class BulletBlocks:
    """
    pybullet API createMultiBody
    """

    def __init__(self, bullet_client, blackboard):
        self._bullet_client = bullet_client
        self._blackboard = blackboard
        self.init_pose = [0.0, 0.0, 1.0]
        self.init_quat = [0.0, 0.0, 0.0, 1.0]
        self.num = self._blackboard.block_num

        self._init_model(self.num)

    def _init_model(self, num):
        visual_file_name = os.path.join(get_urdf_path(), "block/meshes/base_link.STL")
        visual_shape = self._bullet_client.createVisualShape(
            shapeType=self._bullet_client.GEOM_MESH,
            fileName=visual_file_name,
            rgbaColor=[211 / 255, 211 / 255, 211 / 255, 1],
        )
        collision_shape = self._bullet_client.createCollisionShape(
            shapeType=self._bullet_client.GEOM_BOX,
            halfExtents=[
                self._blackboard.BLOCK_LENGTH / 2,
                self._blackboard.BLOCK_LENGTH / 2,
                self._blackboard.BLOCK_HEIGHT / 2,
            ],
        )

        position = [[0, 0, 0] for _ in range(num)]
        self.ids = self._bullet_client.createMultiBody(
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            batchPositions=position,
        )

        for id_ in self.ids:
            self._bullet_client.setCollisionFilterGroupMask(
                id_, -1, collisionFilterGroup=3, collisionFilterMask=3
            )
