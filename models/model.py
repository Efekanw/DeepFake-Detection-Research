from models.ViT.cross_efficient_vit import CrossEfficientViT
from models.Selim.selim_efficient import DeepFakeClassifier


def model(model_name, config):
    """
    Dynamic model function
    :param model_name: model name
    :param config: model config file path
    :return: model
    """
    model_name = model_name
    config = config
    if model_name == 'cross_efficient_vit':
        return CrossEfficientViT(config=config)
    elif model_name == 'selim_efficient':
        return DeepFakeClassifier(config=config)