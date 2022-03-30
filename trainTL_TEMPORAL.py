from matplotlib.pyplot import axis
import numpy as np
import argparse
import torch
from src.model.modelTrafficLightLSTM import TrafficLightNet_64x32_LSTM, TrafficLightNet_128x128_LSTM
from src.model.resnet18LSTM import ResNetLSTM, BasicBlock
from torch import nn, optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import os
from sklearn.metrics import f1_score, confusion_matrix
from src.utils.plotcm import plot_confusion_matrix
from src.dataset.custom_dataset import CustomImageDataset, SequenceDataset
import glob2
import albumentations
import albumentations.pytorch
import random

from src.utils.image_utils import save_img
from src.model.TSM_model import TSN
from src.model.resnet18 import ResNet18
from src.utils.utils import AverageMeter, accuracy

_class_names_id = {0:'Green', 1:'Green_Left', 2:'Red', 3:'Red_Left', 4:'Red_Yellow', 5:'Yellow', 6:'Yellow_Warning'}

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def run_epoches(epochs, model, criterion, optimizer, train_loader, test_loader, patience, checkpoint, device, save_images, args):
    wait = 0
    valid_loss_min = np.Inf
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()
    accuracy_min = 0.0
    best_epoch = 0
    pbar = tqdm(range(epochs), desc="train", mininterval=0.01)
    for epoch in pbar:
        # Train model
        model.train()
        for iter, data in enumerate(train_loader):
            images = data[0].to(device, dtype=torch.float)
            labels = data[1].to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(prec1.item(), images.size(0))
            top5.update(prec5.item(), images.size(0))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if save_images and iter < 10:
                _predict = torch.argmax(outputs.data, dim = 1)
                _images_cpu = images.detach().cpu().numpy()
                _labels_cpu = _predict.detach().cpu().numpy()
                
                for _image, _label in zip(_images_cpu, _labels_cpu):
                    if _image.shape[0] == 10:
                        temp0 = np.concatenate((_image[0]*255,_image[1]*255, _image[2]*255,_image[3]*255,_image[4]*255,_image[5]*255,
                           _image[6]*255,_image[7]*255,_image[8]*255,_image[9]*255), axis=1)
                    elif _image.shape[0] == 8:
                        temp0 = np.concatenate((_image[0]*255,_image[1]*255, _image[2]*255,_image[3]*255,_image[4]*255,_image[5]*255,
                        _image[6]*255,_image[7]*255), axis=1)
                    else: 
                        temp0 = np.concatenate((_image[0]*255,_image[1]*255, _image[2]*255,_image[3]*255), axis=1)
                    temp0 = temp0.transpose(1,2,0)
                    class_name = _class_names_id[_label]
                    save_img(os.path.join('./work/', 'result_img','train', str(epoch) + '/batches_'+ str(iter) + '_' + str(class_name) + '.jpg'),temp0.astype(np.uint8))
        
        torch.save(model.state_dict(), './work/weights/trafficLight_model.pt')

        ### Calidate the model
        model.eval()  ### eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        count = 0
        with torch.no_grad():
            correct = 0
            total = 0
            f1_score_epoch = 0
            _labels = [0, 1, 2, 3, 4, 5, 6]
            conf_matrix = np.zeros((7, 7))
            valid_losses = AverageMeter()
            top1_val = AverageMeter()
            top5_val = AverageMeter()

            for iter, data in enumerate(test_loader):
                images = data[0].to(device, dtype=torch.float)
                labels = data[1].to(device)

                outputs = model(images)

                valid_loss = criterion(outputs, labels)

                # measure accuracy and record loss
                _prec1, _prec5 = accuracy(outputs.data, labels, topk=(1, 5))
                valid_losses.update(valid_loss.item(), images.size(0))
                top1_val.update(_prec1.item(), images.size(0))
                top5_val.update(_prec5.item(), images.size(0))

                predicted = torch.argmax(outputs.data, dim = 1)
                total += len(labels)
                correct += (predicted == labels).sum().item()

                # Calculate F1-score
                f1_score_epoch += f1_score(predicted.cpu().numpy(), labels.cpu().numpy(), average = 'weighted')
                count += 1

                # Calculate the confusion matrix
                conf_matrix += confusion_matrix(y_true = labels.cpu().numpy(), y_pred = predicted.cpu().numpy(), labels = _labels)

                if save_images and iter < 10:
                    _predicted = torch.argmax(outputs.data, dim = 1)
                    _images_cpu = images.detach().cpu().numpy()
                    _labels_cpu = _predicted.detach().cpu().numpy()
                    
                    for _idx, (_image, _label) in enumerate(zip(_images_cpu, _labels_cpu)):
                        if _image.shape[0] == 10:
                            temp0 = np.concatenate((_image[0]*255,_image[1]*255, _image[2]*255,_image[3]*255,_image[4]*255,_image[5]*255,
                            _image[6]*255,_image[7]*255,_image[8]*255,_image[9]*255), axis=1)
                        elif _image.shape[0] == 8:
                            temp0 = np.concatenate((_image[0]*255,_image[1]*255, _image[2]*255,_image[3]*255,_image[4]*255,_image[5]*255,
                            _image[6]*255,_image[7]*255), axis=1)
                        else: 
                            temp0 = np.concatenate((_image[0]*255,_image[1]*255, _image[2]*255,_image[3]*255), axis=1)

                        temp0 = temp0.transpose(1,2,0)
                        class_name = _class_names_id[_label]
                        save_img(os.path.join('./work/', 'result_img','valid', str(epoch) + '/batches_'+ str(_idx) + '_' + str(class_name) + '.jpg'),temp0.astype(np.uint8))
            
            ### Save model if validation loss has decreased
            ### if valid_loss <= valid_loss_min:
            if accuracy_min <= (f1_score_epoch/count):
                #print(f"- Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model...")
                model_save_file = "./work/weights/best_{}_{:.3f}.pt".format(args.model_type, f1_score_epoch/count)
                torch.save(model.state_dict(), model_save_file)
                valid_loss_min = valid_loss
                accuracy_min = (f1_score_epoch / count)
                wait = 0

                label_classes = ['Green','Green_Left','Red','Red_Left','Red_Yellow','Yellow','Yellow_Warning']
                conf_plt = plot_confusion_matrix(conf_matrix, label_classes)
                conf_plt.savefig('./work/Confusion_Matrix_Result_with_TEMPORAL_Data.png')
                conf_plt.close()

                best_epoch = epoch
                best_valid_loss = valid_loss
                best_f1_score = accuracy_min
            # Early stopping
            else:
                wait += 1
                if wait >= patience:
                    print(f"Terminated Training for Early Stopping at Epoch {epoch+1}")
                    print('Best epoch: {}, best f-1 score: {:2.3}, best val loss: {:2.3}'.format(best_epoch, best_f1_score, best_valid_loss))
                    return

        #print('Test Accuracy of the model on the {} test images: {:2.3}%, f1-score {:1.3}'.format(total, 100 * correct / total, f1_score_epoch / count))
        pbar.set_postfix({'Epoch':epoch, 'Train/loss': losses.avg, 'Train/TOP-1': top1.avg, 'Val/accuracy':100 * correct / total, 'Val/f-1 score': f1_score_epoch / count, 'Val/TOP-1': top1_val.avg})

    print('Best epoch: {}, best f-1 score: {:2.3}, best val loss: {:2.3}'.format(best_epoch, best_f1_score, best_valid_loss))

def arg_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='Traffic light recognition training script')
    parser.add_argument('--train_path', type=str, default='/dataset/Traffic_Light_Dataset_KETI/TL_KETI_bbox_temporal/train/', metavar='D',
                        help="Train folder where data is located.")
    parser.add_argument('--valid_path', type=str, default='/dataset/Traffic_Light_Dataset_KETI/TL_KETI_bbox_temporal/valid/', metavar='D',
                        help="Valid folder where data is located.")
    parser.add_argument('--model_type', type=str, default='TSM', metavar='D',
                        help="conv_64x32, conv_128x128, resnetLSTM, TSN, TSM")
    parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                        help='Input batch size for training (default: 64)')
    parser.add_argument('--num_workers', type=int, default=2, metavar='W',
                        help='How many subprocesses to use for data loading (default: 0)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='Number of epochs to train (default: 100)')
    parser.add_argument('--patience', type=int, default=15, metavar='P',
                        help='Number of epochs with no improvement after which training will be stopped (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=2021, metavar='S',
                        help='Random seed (default: 2021)')
    parser.add_argument('--nclass', type=int, default=7, metavar='S',
                        help='number of traffic light class (default: 7)')
    parser.add_argument('--checkpoint', type=str, default='trafficLight_model_64_32_best.pt', metavar='M',
                        help='checkpoint file name (default: model.pt)')
    parser.add_argument('--num_frames', type=int, default=8, metavar='C',
                        help='Temporal sequence frame number')
    parser.add_argument('--save_images', type=bool, default=True, metavar='C',
                        help='Save result image in ./work/result_img/')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Get arguments
    args = arg_parser()

    ### Set random seed 
    set_random_seed(args.seed)

    ### Define train and test loader
    trainVideoFolder = sorted(glob2.glob(args.train_path + "*"))
    validVideoFolder = sorted(glob2.glob(args.valid_path + "*"))

    trainVideo = []
    validVideo = []
    for i in range(len(trainVideoFolder)):
        trainVideo.append(trainVideoFolder[i])

    for i in range(len(validVideoFolder)):
        validVideo.append(validVideoFolder[i])

    albumentations_train = albumentations.Compose([
        albumentations.Resize(32 , 64), 
        albumentations.OneOf([
                        albumentations.MotionBlur(p=1),
                        albumentations.OpticalDistortion(p=1),
                        albumentations.GaussNoise(p=1)                 
        ], p=1),
        albumentations.Blur(p=0.1),
        albumentations.CLAHE(p=0.1),
        albumentations.RandomBrightnessContrast(p=0.2),
        #albumentations.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.2),
        albumentations.pytorch.transforms.ToTensorV2()])

    albumentations_valid = albumentations.Compose([
        albumentations.Resize(32 , 64), 
        albumentations.pytorch.transforms.ToTensorV2()])

    trainDataset = SequenceDataset(trainVideo, max_len = args.num_frames, transform=albumentations_train)
    validDataset = SequenceDataset(validVideo, max_len = args.num_frames, transform=albumentations_valid)
    trainLoader = DataLoader(trainDataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    validLoader = DataLoader(validDataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    if args.model_type == 'conv_64x32':
        model = TrafficLightNet_64x32_LSTM(nclasses=args.nclass).to(device)  
    elif args.model_type == 'conv_128x128':
        model = TrafficLightNet_128x128_LSTM(nclasses=args.nclass).to(device)  
    elif args.model_type == 'resnetLSTM':
        model = ResNetLSTM(BasicBlock, [2, 2, 2, 2], num_classes = args.nclass).to(device)
    elif args.model_type == 'TSN':
        model = TSN(num_class = args.nclass, num_segments = args.num_frames, modality = 'RGB', base_model='resnet18', is_shift = False).to(device)
    elif args.model_type == 'TSM':
        model = TSN(num_class = args.nclass, num_segments = args.num_frames, modality = 'RGB', base_model='resnet18', is_shift = True).to(device)
        ### Temporal data must set to 8 in TSN model.
        ### For various number of temporal data setting need to check Fold div in TSM
        args.num_frames = 8

    ### Optimizer and Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    run_epoches(args.epochs, model, criterion, optimizer,
        trainLoader, validLoader, args.patience, args.checkpoint, device, save_images=args.save_images, args=args)