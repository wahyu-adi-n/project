# from models.convnext import ConvNext
from models.efficientnet import EfficientNetB0
# from models.ensemble_effiecientnet_vit import EnsembleEfficientNetViT
# from models.ensemble_vit_convnext import EnsembleConvNextVIT
# from models.vision_transformer import VIT
from torchvision import models

def create_model(model_name: str, **kwargs):
    """
    Create model based on model_name.
    Args:
        model_name (str): model name
    Return:
        model
    """
    model = models.efficientnet_b0(pretrained=True) 
    if model_name == 'ensemble_convnext_vit':
        # model = EnsembleConvNextVIT(**kwargs)
        pass
    elif model_name == 'ensemble_efficientnet_vit':
        # model = EnsembleEfficientNetViT(**kwargs)
        pass
    elif model_name == 'vit':
        # model = VIT(**kwargs)
        pass
    elif model_name == 'efficientnet_b0':
        model = EfficientNetB0(**kwargs)
    elif model_name == 'convnext':
        # model = ConvNext(**kwargs)
        pass
    else:
        raise NotImplementedError
    return model
