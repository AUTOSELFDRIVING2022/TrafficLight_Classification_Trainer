import torch
import torch.nn as nn
import numpy as np
#from monai.metrics.utils import get_mask_edges, get_surface_distance
from scipy.spatial.distance import directed_hausdorff
import segmentation_models_pytorch as smp

def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice

def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou

JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
DiceLoss    = smp.losses.DiceLoss(mode='multilabel')
FocalLoss   = smp.losses.FocalLoss(mode='multilabel')
LovaszLoss  = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)

class BCEandTversky(nn.Module):
    def __init__(self):
        super(BCEandTversky, self).__init__()
        self.w1 = 0.5
        self.w2 = 0.5
    def forward(self, y_pred, y_true):
        return self.w1*BCELoss(y_pred, y_true) + self.w2*TverskyLoss(y_pred, y_true)
    
class FocalandTversky(nn.Module):
    def __init__(self):
        super(FocalandTversky, self).__init__()
        self.w1 = 0.5
        self.w2 = 0.5
    def forward(self, y_pred, y_true):
        return self.w1*FocalLoss(y_pred, y_true) + self.w2*TverskyLoss(y_pred, y_true)

def get_criterion(cfg):
    if cfg.losses.name == 'CE':
        return nn.CrossEntropyLoss()
    if cfg.losses.name == 'BCEandTversky':
        return BCEandTversky()
    elif cfg.losses.name == 'FocalandTversky':
        return FocalandTversky()
    else: 
        raise NameError('Choose proper model name!!!')

if __name__ == "__main__":
    true = np.array([1.0, 1.2, 1.1, 1.4, 1.5, 1.8, 1.9])
    pred = np.array([1.0, 1.2, 1.1, 1.4, 1.5, 1.8, 1.9])
    
    #loss = criterion(pred, true)
    #print(loss)