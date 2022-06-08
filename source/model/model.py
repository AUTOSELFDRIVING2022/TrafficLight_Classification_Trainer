import segmentation_models_pytorch as smp
from .modelTrafficLight import TrafficLightNet_64x32_noSTN, TrafficLightNet_64x32_coordConv
from .modelTrafficLightLSTM import TrafficLightNet_128x128_LSTM, TrafficLightNet_64x32_LSTM
from .resnet18LSTM import ResNetLSTM, BasicBlock
from .resnet18 import ResNet18
from .TSM_model import TSN
import torch

def get_model(cfg):
    if cfg.train_config.model_type == 'single_frame':
        return build_model_single(cfg)
    elif cfg.train_config.model_type == 'temporal_frame':
        return build_model_temporal(cfg)

def build_model_single(cfg):
    if cfg.model.name == 'conv_64x32':
        model = TrafficLightNet_64x32_noSTN(cfg.dataset.num_classes) 
    elif cfg.model.single_frame.name == 'conv_64x32_coordConv':
        model = TrafficLightNet_64x32_coordConv(cfg.dataset.num_classes)
    elif cfg.model.single_frame.name == 'resnet18_64x64':
        model = ResNet18(cfg.dataset.num_classes)
    else: 
        raise NameError('Choose proper model name!!!')
    
    model.to(cfg.train_config.device)
    return model

def build_model_temporal(cfg):
    if cfg.model.temporal_frame.name == 'conv_64x32':
        model = TrafficLightNet_64x32_LSTM(nclasses=cfg.dataset.num_classes)
    elif cfg.model.temporal_frame.name == 'conv_128x128':
        model = TrafficLightNet_128x128_LSTM(nclasses=cfg.dataset.num_classes)
    elif cfg.model.temporal_frame.name == 'resnetLSTM':
        model = ResNetLSTM(BasicBlock, [2, 2, 2, 2], num_classes = cfg.dataset.num_classes)
    elif cfg.model.temporal_frame.name == 'TSN':
        model = TSN(num_class = cfg.dataset.num_classes, num_segments = cfg.dataset.num_frames, modality = 'RGB', base_model='resnet18', is_shift = False)
    elif cfg.model.temporal_frame.name == 'TSM':
        model = TSN(num_class = cfg.dataset.num_classes, num_segments = cfg.dataset.num_frames, modality = 'RGB', base_model='resnet18', is_shift = True)
        ### Temporal data must set to 8 in TSN model.
        ### For various number of temporal data setting need to check Fold div in TSM
        cfg.dataset.num_frames = 8
    else: 
        raise NameError('Choose proper model name!!!')
    model.to(cfg.train_config.device)
    return model

if __name__ == "__main__":
    cfg = 'None'
    modelTrain = get_model(cfg)
    print(modelTrain)