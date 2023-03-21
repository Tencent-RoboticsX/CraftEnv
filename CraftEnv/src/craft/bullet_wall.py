import os
from craft import get_urdf_path


class BulletWalls:
    """
    pybullet API createMultiBody
    """

    def __init__(self, bullet_client, blackboard):
        self._bullet_client = bullet_client
        self._blackboard = blackboard
        self.init_pose = [0.0, 0.0, 1.0]
        self.init_quat = [0.0, 0.0, 0.0, 1.0]
        self.num = self._blackboard.wall_num

        self._init_model(self.num)

    def _init_model(self, num):
        visual_file_name = os.path.join(get_urdf_path(), "wall/meshes/base_link.STL")
        visual_shape = self._bullet_client.createVisualShape(
            shapeType=self._bullet_client.GEOM_MESH,
            fileName=visual_file_name,
            rgbaColor=[211 / 255, 211 / 255, 211 / 255, 0.1],
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
