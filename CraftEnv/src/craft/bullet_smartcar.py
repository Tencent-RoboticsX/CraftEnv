import os
from craft import get_urdf_path


class BulletSmartcar:
    def __init__(self, bullet_client):
        self._bullet_client = bullet_client
        self.init_pose = [0.0, 0.0, 1.0]
        self.init_quat = [0.0, 0.0, 0.0, 1.0]
        self._init_model(self.init_pose, self.init_quat)

    def _init_model(self, init_pose, init_quat):
        self.robot_id = self._bullet_client.loadURDF(
            os.path.join(get_urdf_path(), "smartcar/smartcar.urdf"),
            flags=(self._bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            globalScaling=1,
        )
        for i in range(-1, self._bullet_client.getNumJoints(self.robot_id)):
            self._bullet_client.setCollisionFilterGroupMask(
                self.robot_id, i, collisionFilterGroup=3, collisionFilterMask=3
            )
        self._bullet_client.changeVisualShape(self.robot_id, 0, rgbaColor=[211 / 255, 211 / 255, 211 / 255, 1])
        self._bullet_client.changeVisualShape(self.robot_id, 1, rgbaColor=[211 / 255, 211 / 255, 211 / 255, 1])
