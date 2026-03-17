from torch.nn import Module
from torch.nn import Identity
from os.path import join, abspath
import sys
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../../../'))
sys.path.append(project_path)

import models.utils.TIP_utils.resnets as resnets

def torchvision_ssl_encoder(
    name: str,
    pretrained: bool = False,
    return_all_feature_maps: bool = False,
) -> Module:
    pretrained_model = getattr(resnets, name)(pretrained=pretrained, return_all_feature_maps=return_all_feature_maps)
    pretrained_model.fc = Identity()
    return pretrained_model
