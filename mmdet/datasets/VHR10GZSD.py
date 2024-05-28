# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .xml_style import XMLDataset

import numpy as np

@DATASETS.register_module()
class VHR10GZSDDataset(XMLDataset):
    "Dataset for NWPU VHR-10"

    METAINFO = {
        'classes':
        ('airplane', 'ship', 'storagetank', 'tenniscourt', 'basketballcourt', 'baseballfield', 'groundtrackfield', 'harbor', 'vehicle', 
         'bridge'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
