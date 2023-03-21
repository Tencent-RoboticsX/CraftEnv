from abc import ABC, abstractmethod
from enum import IntEnum, unique, auto


@unique
class ObjType(IntEnum):
    Undefined = -1
    Air = auto()
    Ground = auto()
    Wall = auto()
    Block = auto()
    Flag = auto()
    FoldedSlope = auto()
    FoldedSlopeGear = auto()
    UnfoldedSlopeBody = auto()
    UnfoldedSlopeFoot = auto()


class WorldObj(ABC):
    """
    Base class for grid world objects
    """

    @abstractmethod
    def __init__(self):
        self.can_lift = False
        self.can_fold = False
        self.can_unfold = False
        self.can_stand = False
        self.near_unfold_slope_body = False
        self.near_blow_unfold_slope_foot = False
        """
        the obj id on the WorldObj, -1 means there is nothing, -2 means there is something,
        0, 1, 2... means there is a smartcar on it and the number represent the smartcar's id
        """
        self.obj_on_it = -1
        self.type = ObjType.Undefined


class Air(WorldObj):
    def __init__(self):
        super().__init__()
        self.can_lift = False
        self.can_stand = False
        self.type = ObjType.Air


class Ground(WorldObj):
    def __init__(self):
        super().__init__()
        self.can_stand = True
        self.type = ObjType.Ground


class Wall(WorldObj):
    def __init__(self):
        super().__init__()
        self.type = ObjType.Wall
        self.can_lift = False
        self.can_stand = False


class Block(WorldObj):
    def __init__(self):
        super().__init__()
        self.can_lift = True
        self.can_stand = True
        self.type = ObjType.Block


class Flag(WorldObj):
    def __init__(self):
        super().__init__()
        self.can_lift = True
        self.type = ObjType.Flag


class FoldedSlope(WorldObj):
    """
     ↑        1
    ← →  ↔  2   0
     ↓        3

    body-foot
    yaw = 0

    foot
    |
    body
    yaw = np.pi * 0.5

    foot-body
    yaw = np.pi * 1

    body
    |
    foot
    yaw = np.pi * 1.5
    """

    def __init__(self, yaw):
        super().__init__()
        self.can_lift = True
        self.yaw = yaw
        self.type = ObjType.FoldedSlope


class FoldedSlopeGear(WorldObj):
    def __init__(self, yaw):
        super().__init__()
        self.yaw = yaw
        self.type = ObjType.FoldedSlopeGear


class UnfoldedSlopeBody(WorldObj):
    """
     ↑        1
    ← →  ↔  2   0
     ↓        3
    """

    def __init__(self, yaw):
        super().__init__()
        self.yaw = yaw
        self.type = ObjType.UnfoldedSlopeBody


class UnfoldedSlopeFoot(WorldObj):
    """
     ↑        1
    ← →  ↔  2   0
     ↓        3
    """

    def __init__(self, yaw):
        super().__init__()
        self.yaw = yaw
        self.type = ObjType.UnfoldedSlopeFoot
