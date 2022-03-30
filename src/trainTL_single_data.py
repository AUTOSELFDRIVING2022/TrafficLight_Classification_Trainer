import numpy as np
import argparse
import torch
from src.model.modelTrafficLight import TrafficLightNet_64x32_coordConv
from torch import nn, optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import os
from sklearn.metrics import f1_score, confusion_matrix
from utils.plotcm import plot_confusion_matrix
import cv2
from src.dataset.custom_dataset import CustomImageDataset

label_classes = ['Green', 'Green Left', 'Red_Left', 'Red', 'Yellow', 'Off', 'Other']

def run_epoches(epochs, model, criterion, optimizer, train_loader, test_loader, patience, checkpoint, device):
    wait = 0
    valid_loss_min = np.Inf
    accuracy_min = 0.0
    for epoch in range(epochs):
        for data in train_loader:
            images = data['image'].to(device, dtype=torch.float)
            labels = data['label'].to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))
        torch.save(model.state_dict(), 'trafficLight_model.pt')

        # Test the model
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        count = 0
        with torch.no_grad():
            correct = 0
            total = 0
            f1_score_epoch = 0
            _labels = [0,1,2,3,4,5,6]
            conf_matrix = np.zeros((7,7))
            for data in test_loader:
                images = data['image'].to(device, dtype=torch.float)
                labels = data['label'].to(device)

                outputs = model(images)

                valid_loss = criterion(outputs, labels)

                predicted = torch.argmax(outputs.data, dim=1)
                total += len(labels)
                correct += (predicted == labels).sum().item()
                # Calculate F1-score
                f1_score_epoch += f1_score(predicted.cpu().numpy(), labels.cpu().numpy(), average='weighted')
                count += 1

                # Calculate the confusion matrix
                conf_matrix += confusion_matrix(y_true=labels.cpu().numpy(), y_pred=predicted.cpu().numpy(), labels=_labels)
            
            #label_classes_rev = ['Other', 'Red', 'Off', 'Yellow', 'Red_Left', 'Green_Left', 'Green']
            #label_classes.reverse()

            # save model if validation loss has decreased
            #if valid_loss <= valid_loss_min:
            if accuracy_min <= (f1_score_epoch/count):
                print(f"- Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model...")
                model_save_file = "{}_{:.3f}.pt".format(checkpoint[:-3],f1_score_epoch/count)
                torch.save(model.state_dict(), model_save_file)
                valid_loss_min = valid_loss
                accuracy_min = (f1_score_epoch/count)
                wait = 0

                conf_plt = plot_confusion_matrix(conf_matrix, label_classes)
                conf_plt.savefig('confusion_matrix_ketiNEW.png')
                conf_plt.close()
            # Early stopping
            else:
                wait += 1
                if wait >= patience:
                    print(f"Terminated Training for Early Stopping at Epoch {epoch+1}")
                    return

        print('Test Accuracy of the model on the {} test images: {:2.3}%, f1-score {:1.3}'.format(total, 100 * correct / total, f1_score_epoch / count))

def arg_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='Traffic light recognition training script')
    parser.add_argument('--train-path', type=str, default='/dataset/Traffic_Light_Dataset_KETI4/train/', metavar='D',
                        help="Train folder where data is located.")
    parser.add_argument('--valid-path', type=str, default='/dataset/Traffic_Light_Dataset_KETI4/test/', metavar='D',
                        help="Valid folder where data is located.")
    parser.add_argument('--class-count', type=int, default=20000, metavar='C',
                        help='Each class will have this number of samples after extension and balancing (default: 10k)')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='Input batch size for training (default: 64)')
    parser.add_argument('--num-workers', type=int, default=8, metavar='W',
                        help='How many subprocesses to use for data loading (default: 0)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='Number of epochs to train (default: 100)')
    parser.add_argument('--patience', type=int, default=15, metavar='P',
                        help='Number of epochs with no improvement after which training will be stopped (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='Random seed (default: 1)')
    parser.add_argument('--nclass', type=int, default=7, metavar='S',
                        help='number of traffic light class (default: 7)')
    parser.add_argument('--checkpoint', type=str, default='trafficLight_model_64_32_best_lab_coordconv.pt', metavar='M',
                        help='checkpoint file name (default: model.pt)')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    return args

def set_data_loader(args):
    train_data_set = CustomImageDataset(data_set_path=args.train_path)
    train_loader = DataLoader(train_data_set, batch_size=args.batch_size, shuffle=True)

    test_data_set = CustomImageDataset(data_set_path=args.valid_path)
    test_loader = DataLoader(test_data_set, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Get arguments
    args = arg_parser()
    
    ### Define train and test loader
    train_loader, test_loader = set_data_loader(args)

    ### DDRNet_23_slim
    #model = get_model(_planes=3, _last_planes=args.nclass).to(device)
    ### Define Model
    #model = TrafficLightNet_64x64_noSTN(args.nclass).to(device)
    
    #model = TrafficLightNet_64x64_coordConv(args.nclass).to(device)
    model = TrafficLightNet_64x32_coordConv(args.nclass).to(device)
    
    #model = TrafficLightNet_32x32_noSTN(args.nclass).to(device)
    #model = TrafficLightNet_64x32_noSTN(args.nclass).to(device)
    
    #model = TrafficLightNet_32x32_coordConv(n_class).to(device)
    #model = TrafficLightNet(n_class).to(device)
    #model = TrafficLightNet_32x32(n_class).to(device)
    #model = TrafficLightNet_32x32_noSTN(n_class).to(device)
    
    ### Optimizer and Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    run_epoches(args.epochs, model, criterion, optimizer,
        train_loader, test_loader, args.patience, args.checkpoint, device)