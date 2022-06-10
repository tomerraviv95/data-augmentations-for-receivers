from enum import Enum

HALF = 0.5


class Phase(Enum):
    TRAIN = 'train'
    TEST = 'test'


class ChannelModes(Enum):
    SISO = 'SISO'
    MIMO = 'MIMO'

class DetectorType(Enum):
    black_box = 'black_box'
    model = 'model'
