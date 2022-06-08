import numpy as np
import pandas as pd

from glob import glob
import os, shutil
from tqdm import tqdm
import time
import copy
import joblib
from collections import defaultdict
import gc
from IPython import display as ipd

# PyTorch 
import torch

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torch.cuda import amp

import timm

#import rasterio
from joblib import Parallel, delayed
from sklearn.metrics import f1_score, confusion_matrix

import hydra
from hydra_plugins.hydra_optuna_sweeper.config import OptunaSweeperConf
from omegaconf import DictConfig, OmegaConf, ListConfig

from source.utils.config import set_seed
from source.model.model import get_model
from source.losses import dice_coef, get_criterion, iou_coef
#from source.dataloader import prepare_loaders, save_batch, plot_batch
from source.optimizer import get_optimizer, get_scheduler
from source.dataset.custom_dataset import get_dataset
from source.utils.utils import AverageMeter, accuracy
from source.utils.plotcm import plot_confusion_matrix

# For colored terminal text
from colorama import Fore, Back, Style
c_  = Fore.GREEN
sr_ = Style.RESET_ALL

import warnings
warnings.filterwarnings("ignore")

#Wandb and mlflow
import wandb
import mlflow

try:
    wandb.login(key='0b0a03cb580e75ef44b4dff7f6f16ce9cfa8a290')
    anonymous = None
except:
    anonymous = "must"
    print('To use your W&B account,\nGo to Add-ons -> Secrets and provide your W&B access token. Use the Label name as WANDB. \nGet your W&B access token from here: https://wandb.ai/authorize')
            
def train_one_epoch(cfg, model, optimizer, scheduler, criterion, dataloader, device, epoch):
    model.train()
    
    scaler = amp.GradScaler()
    max_norm = 5.0
    
    train_top1 = AverageMeter()
    train_top5 = AverageMeter()
    f1_scores = AverageMeter()
    train_loss = AverageMeter()
            
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train:')
    
    for step, (data) in pbar:         
        images = data['image'].to(device, dtype=torch.float)
        labels  = data['label'].to(device)
        
        batch_size = images.size(0)
        
        ###Use Unscaled Gradiendts instead of 
        ### https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
        with amp.autocast(enabled=True):
            outputs = model(images)
            loss   = criterion(outputs, labels)    
        scaler.scale(loss).backward()
        
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        if scheduler is not None:
            scheduler.step()
                
        # measure accuracy and record loss
        predicted = torch.argmax(outputs.data, dim=1)
        f1_scores.update(f1_score(predicted.cpu().numpy(), labels.cpu().numpy(), average='weighted'), batch_size)
        
        prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
        train_top1.update(prec1.item(), batch_size)
        train_top5.update(prec5.item(), batch_size)
            
        train_loss.update(loss.item(), batch_size)
        #running_loss += (loss.item() * batch_size)
        #dataset_size += batch_size
        
        #epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        
        pbar.set_postfix(epoch=f'{epoch}',
                        train_loss=f'{train_loss.avg:0.4f}',
                        lr=f'{current_lr:0.5f}', 
                        top1=f'{train_top1.avg:0.2f}',
                        f1_score=f'{f1_scores.avg:0.2f}',
                        gpu_mem=f'{mem:0.2f} GB')
        
        #if cfg.train_config.debug and step < 30:
            # _imgs  = images.cpu().detach()
            
            # _outputs = (nn.Sigmoid()(outputs)>0.5).double()
            # _outputs = _outputs.cpu().detach()
            
            # _labels = labels.cpu().detach()
            #_y_pred = torch.mean(torch.stack(_y_pred, dim=0), dim=0).cpu().detach()
            
            #plot_batch(imgs=_imgs, pred_msks=_y_pred, gt_msks=_masks, size=5, step = step, epoch = epoch, mode = 'train')
            #save_batch(_imgs, _y_pred, size = 5, step = step, epoch = epoch, mode = 'train')
    return train_loss.avg, train_top1.avg
    
@torch.no_grad()
def valid_one_epoch(cfg, model, dataloader, criterion, device, epoch, optimizer):
    model.eval()
    
    val_top1 = AverageMeter()
    val_top5 = AverageMeter()
    val_loss = AverageMeter()
    f1_scores = AverageMeter()
    conf_matrix = np.zeros((7, 7))
    
    _total_itr = 0
    _acc = 0
    _labels = [0, 1, 2, 3, 4, 5, 6]
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
    
    for step, (data) in pbar:         
        images = data['image'].to(device, dtype=torch.float)
        labels  = data['label'].to(device)
        
        batch_size = images.size(0)
        
        outputs  = model(images)
        loss    = criterion(outputs, labels)
        
        ### Loss
        val_loss.update(loss.item(), batch_size)
        
        ### Accuracy
        predicted = torch.argmax(outputs.data, dim = 1)
        _prec1, _prec5 = accuracy(outputs.data, labels, topk=(1, 5))
        val_top1.update(_prec1.item(), batch_size)
        val_top5.update(_prec5.item(), batch_size)
        f1_scores.update(f1_score(predicted.cpu().numpy(), labels.cpu().numpy(), average = 'weighted'), batch_size)
        _acc += (predicted == labels).sum().item() 
        _total_itr += batch_size
        _acc_step = _acc / _total_itr
        
        conf_matrix += confusion_matrix(y_true = labels.cpu().numpy(), y_pred = predicted.cpu().numpy(), labels = _labels)
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss=f'{val_loss.avg:0.4f}',
                        f1_scores=f'{f1_scores.avg:0.2f}',
                        acc=f'{_acc_step:0.2f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_memory=f'{mem:0.2f} GB')
        
        #if cfg.train_config.debug and step < 0:
            # Calculate the confusion matrix
    return val_loss.avg, f1_scores.avg, val_top1.avg, _acc_step, conf_matrix

def run_training(cfg, model, optimizer, scheduler, criterion, device, num_epochs, train_loader, valid_loader, run_log_wandb):
    
    # To automatically log gradients
    wandb.watch(model, log_freq=100)
    
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_f1_scores      = -np.inf
    best_epoch     = -1
    best_acc       = -1
    
    # start new run
    mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
    mlflow.set_experiment(cfg.train_config.comment)
    with mlflow.start_run():
        for epoch in range(1, num_epochs + 1): 
            print(f'Epoch {epoch}/{num_epochs}', end='')
            
            train_loss, train_top1 = train_one_epoch(cfg, model, optimizer, scheduler, criterion= criterion,
                                            dataloader=train_loader, 
                                            device=device, epoch=epoch)
            
            val_loss, f1_scores, val_top1, val_acc, conf_mat = valid_one_epoch(cfg, model, valid_loader, criterion,
                                                    device=device, 
                                                    epoch=epoch, optimizer=optimizer)

            # Log the metrics
            wandb.log({"Train Loss": train_loss, 
                    "Train Top1": train_top1,   
                    "Valid Loss": val_loss,
                    "Valid F1 score": f1_scores,
                    "Valid Top1": val_top1,
                    "Valid ACC": val_acc,
                    "LR":scheduler.get_last_lr()[0]})
            
            #Mlflow log
            mlflow.log_metric("Train_loss", train_loss, step=epoch)
            mlflow.log_metric("Train_lr", scheduler.get_last_lr()[0], step=epoch)
            mlflow.log_metric("Train_top1", train_top1, step=epoch)
            mlflow.log_metric("Val_loss", val_loss, step=epoch)
            mlflow.log_metric("Val_F1_score", f1_scores, step=epoch)
            mlflow.log_metric("Val_Top1", val_top1, step=epoch)
            mlflow.log_metric("Val_ACC", val_acc, step=epoch)
            
            # deep copy the model
            if f1_scores >= best_f1_scores:
                print(f"{c_}Valid Score Improved ({best_f1_scores:0.4f} ---> {f1_scores:0.4f})")
                best_f1_scores    = f1_scores
                best_epoch        = epoch
                best_acc          = val_acc
                run_log_wandb.summary["Best F1"]    = best_f1_scores
                run_log_wandb.summary["Best Epoch"]   = best_epoch
                run_log_wandb.summary["Best Acc"]   = best_acc
                #best_model_wts = copy.deepcopy(model.state_dict())

                dirPath = "./run/{}".format(cfg.train_config.comment)
                fold = 0
                PATH = f"best_epoch-{fold:02d}.bin"
                torch.save(model.state_dict(), os.path.join(dirPath,PATH))
                # Save a model file from the current directory
                wandb.save(PATH)
                print(f"Model Saved{sr_}")
                
                label_classes = ['Green', 'Green Left', 'Red_Left', 'Red', 'Yellow', 'Off', 'Other']
                conf_plt = plot_confusion_matrix(conf_mat, label_classes)
                conf_plt.savefig(os.path.join(dirPath,'confusion_matrix_keti_best.png'))
                conf_plt.close()

            #last_model_wts = copy.deepcopy(model.state_dict())
            fold = 0
            PATH = f"last_epoch-{fold:02d}.bin"
            torch.save(model.state_dict(), PATH)
            print(); print()
            
            torch.cuda.empty_cache()
            gc.collect()

        end = time.time()
        time_elapsed = end - start
        print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
        print("Best F1-Score: {:.4f}".format(best_f1_scores))
    return model

@hydra.main(config_path="conf", config_name="config")
def train(cfg : DictConfig) -> None:
    set_seed()
    
    ### For use original path.
    ### For debug...
    #currPath = os.getcwd()
    #os.chdir(currPath)
    #print(currPath)
    # org_cwd = hydra.utils.get_original_cwd()
    # print(org_cwd)
    
    dirPath = "./run/{}".format(cfg.train_config.comment)
    if not os.path.isdir(dirPath): 
        os.makedirs(dirPath)
    
    model = get_model(cfg)
    train_loader, valid_loader  = get_dataset(cfg) 
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)
    criterion = get_criterion(cfg)
    print(cfg.train_config.comment)
    
    run_log_wandb = wandb.init(project='Traffic-Light-Recognization', 
                    config={k:v for k, v in dict(cfg).items() if '__' not in k},
                    anonymous=anonymous,
                    name=f"dim-{cfg.train_config.img_size[0]}x{cfg.train_config.img_size[1]}|model-{cfg.model.name}",
                    group=cfg.train_config.comment,
                    )
    model = run_training(cfg, model, optimizer, scheduler, criterion= criterion,
                                device=cfg.train_config.device,
                                num_epochs=cfg.train_config.epochs,
                                train_loader=train_loader,
                                valid_loader=valid_loader,
                                run_log_wandb=run_log_wandb)
    run_log_wandb.finish()

if __name__ == "__main__":
    train()