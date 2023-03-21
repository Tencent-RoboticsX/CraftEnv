import os
from craft import get_urdf_path


class BulletFlag:
    def __init__(self, bullet_client):
        self._bullet_client = bullet_client
        self.init_pose = [0.0, 0.0, 1.0]
        self.init_quat = [0.0, 0.0, 0.0, 1.0]
        self._init_model(self.init_pose, self.init_quat)

    def _init_model(self, init_pose, init_quat):
        robot_path = os.path.join(get_urdf_path(), "flag/block.urdf")
        self.robot_id = self._bullet_client.loadURDF(
            robot_path,
            init_pose,
            init_quat,
            flags=(self._bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )

        for i in range(self._bullet_client.getNumJoints(self.robot_id)):
            self._bullet_client.changeVisualShape(
                self.robot_id, i, rgbaColor=[1, 0, 0, 1]
            )

        self._bullet_client.setCollisionFilterGroupMask(
            self.robot_id, -1, collisionFilterGroup=3, collisionFilterMask=3
        )
