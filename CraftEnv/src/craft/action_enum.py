from enum import IntEnum, auto, unique
import numpy as np


ACTION_ARG = [
    None,
    None,
    None,
    None,
    1,  # TURN_LEFT
    -1,  # TURN_RIGHT
    # (x, y)
    np.array((0, 1)),  # MOVE_FORWARD
    np.array((0, -1)),  # MOVE_BACK
    np.array((-1, 0)),  # MOVE_LEFT
    np.array((1, 0)),  # MOVE_RIGHT
    None,  # STOP
    np.array((-1, 1)),  # MOVE_FORWARD_LEFT
    np.array((1, 1)),  # MOVE_FORWARD_RIGHT
    np.array((-1, -1)),  # MOVE_BACK_LEFT
    np.array((1, -1)),  # MOVE_BACK_RIGHT
]


@unique
class ActionEnum(IntEnum):
    """
    ^ y
    |
    |
    |
    o---------> x
    world coordinate

    smartcar action enum

    MOVE_FORWARD:    ^
                     |

    MOVE_BACK:       |
                     v

    MOVE_LEFT:       <--

    MOVE_RIGHT:      -->

    TURN_LEFT       <--
                      |

    TURN_RIGHT        -->
                      |

    LIFT

    DROP

    FOLD

    UNFOLD

    STOP
    """

    LIFT = 0  # 0
    DROP = auto()  # 1
    FOLD = auto()  # 2
    UNFOLD = auto()  # 3
    ROTATE_LEFT = auto()  # 4
    ROTATE_RIGHT = auto()  # 5
    MOVE_FORWARD = auto()  # 6
    MOVE_BACK = auto()  # 7
    MOVE_LEFT = auto()  # 8
    MOVE_RIGHT = auto()  # 9
    STOP = auto()  # 10
    MOVE_FORWARD_LEFT = auto()  # 11
    MOVE_FORWARD_RIGHT = auto()  # 12
    MOVE_BACK_LEFT = auto()  # 13
    MOVE_BACK_RIGHT = auto()  # 14
