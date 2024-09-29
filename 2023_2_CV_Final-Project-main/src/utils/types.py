from enum import Enum
from argparse import Namespace


class DatasetType(Enum):
    CrowdHuman = "CH"
    EuroCityPersons = "ECP"
    Indoor = "IN"
    Combined = "combined"
    Test = "test"


class ResizeInputArgs(Namespace):
    dataset_type: DatasetType


class VisualizeBBoxArgs(Namespace):
    dataset_type: DatasetType
    score: float = 0.5
