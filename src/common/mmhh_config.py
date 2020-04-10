from dataloader.image_list import load_images

from enum import Enum

global_debugging = False


class ImageLossType(Enum):
    DHN = 1
    HashNet = 2


class TestType(Enum):
    Test = 1
    NoFirst = 2
    NoTest = 3


class DistanceType(Enum):
    Hamming = 1
    tSNE = 2
    Cauchy = 3
    Margin1 = 4
    Margin2 = 5
    Metric = 6
    MMHH = 7


class BatchType(Enum):
    PairBatch = 1
    SemiMem = 2
    BatchInitMem = 3
    BatchSelectMem = 4


class SemiInitType(Enum):
    RANDOM = 1
    MODEL = 2


class Mission(Enum):
    Cross_Modal_Transfer = 1
    Hashing = 2
    Cross_Domain_Transfer = 3


class LrScheduleType(Enum):
    Stair_Step = 1
    DANN_INV = 2
    HashNet_Step = 3
    ShapeNet_Step = 4
    TimeSeries_Step = 5


class MarginParams(object):
    def __init__(self, sim_in, dis_in, sim_out=1.0, dis_out=1.0, margin=2.0):
        self.sim_in = sim_in
        self.dis_in = dis_in
        self.sim_out = sim_out
        self.dis_out = dis_out
        self.margin = margin

    @staticmethod
    def from_string_params(param_str: str):
        param_strs = param_str.split(":")
        if len(param_strs) == 2:
            return MarginParams(float(param_strs[0]), float(param_strs[1]))
        if len(param_strs) == 3:
            return MarginParams(float(param_strs[0]), float(param_strs[1]), float(param_strs[2]),
                                float(param_strs[3]))
        if len(param_strs) == 4:
            return MarginParams(float(param_strs[0]), float(param_strs[1]), float(param_strs[2]),
                                float(param_strs[3]))
        if len(param_strs) == 5:
            return MarginParams(float(param_strs[0]), float(param_strs[1]), float(param_strs[2]),
                                float(param_strs[3]), float(param_strs[4]))


data_config = {
    "coco_80": {
        "train": "../data/coco/80_coco/train.txt",
        "database": "../data/coco/80_coco/database.txt",
        "test": "../data/coco/80_coco/test.txt",
        "R": 5000,
        "loader": load_images, "class_num": 1},
    "nuswide_21": {
        "train": "../data/nuswide/nuswide_21/train.txt",
        "database": "../data/nuswide/nuswide_21/database.txt",
        "test": "../data/nuswide/nuswide_21/test.txt",
        "R": 5000,
        "loader": load_images, "class_num": 5},
    "nuswide_81": {
        "train": "../data/nuswide/nuswide_81/train.txt",
        "database": "../data/nuswide/nuswide_81/database.txt",
        "test": "../data/nuswide/nuswide_81/test.txt",
        "R": 5000,
        "loader": load_images, "class_num": 5},
}

tensorboard_interval = 50
