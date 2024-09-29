from utils.types import DatasetType
from utils.consts import *


class DataConfig:
    """
    Data configuration class
    Put the dataset type (CH, ECP, IN) and get the constants of the dataset

    Attributes:
        IMAGE_PATH: path of image files
        LABEL_PATH: path of label files
        RESIZED_OUTPUT_PATH: path of resized image files
        VIZ_BBOX_OUTPUT_PATH: path of visualized bounding box image files
    """

    def __init__(self, dataset_type: DatasetType) -> None:
        assert (
            dataset_type in DatasetType._member_map_.values()
        ), f"robot type should be a member of ({DatasetType._member_names_})."

        if dataset_type == DatasetType.CrowdHuman:
            self.IMAGE_PATH = CROWDHUMAN_IMAGE_PATH
            self.LABEL_PATH = CROWDHUMAN_LABEL_PATH
            self.RESIZED_OUTPUT_PATH = CROWDHUMAN_RESIZED_OUTPUT_PATH
            self.VIZ_BBOX_OUTPUT_PATH = CROWDHUMAN_VIZ_BBOX_OUTPUT_PATH

        elif dataset_type == DatasetType.EuroCityPersons:
            self.IMAGE_PATH = ECP_IMAGE_PATH
            self.LABEL_PATH = ECP_LABEL_PATH
            self.VIZ_BBOX_OUTPUT_PATH = ECP_VIZ_BBOX_OUTPUT_PATH

        elif dataset_type == DatasetType.Combined:
            self.IMAGE_PATH = COMBINE_IMAGE_PATH
            self.LABEL_PATH = COMBINE_LABEL_PATH
            self.VIZ_BBOX_OUTPUT_PATH = COMBINE_VIZ_BBOX_OUTPUT_PATH

        elif dataset_type == DatasetType.Test:
            self.IMAGE_PATH = TEST_IMAGE_PATH
            self.LABEL_PATH = TEST_LABEL_PATH
            self.VIZ_BBOX_OUTPUT_PATH = TEST_VIZ_BBOX_OUTPUT_PATH

        # add other dataset's configuration if they are added

        # ex)
        # elif dataset_type == DatasetType.EuroCityPersons:
