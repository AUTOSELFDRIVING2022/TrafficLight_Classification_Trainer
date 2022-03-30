import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
from torch import nn
from src.model.modelTrafficLight import TrafficSignNet, TrafficSignNetRGB, TrafficLightNet, TrafficLightNet_64x32_noSTN
from dataset.data import get_test_loader, get_test_loaderRGB, get_test_loadersLight
from torchvision.utils import make_grid
from tqdm import tqdm
from sklearn.metrics import f1_score
import cv2
import random
import string
from torch.utils.data import Dataset, DataLoader
from dataset.custom_dataset import CustomImageDataset
import os

def evaluate(model, loss_func, dl):
    model.eval()
    with torch.no_grad():
        for data in dl:
            x = data['image'].to('cuda',dtype=torch.float)
            y = data['label'].to('cuda')
            fname = os.path.basename(data['name'][0])
            output = model(x)
            loss = loss_func(output, y)
            pred = torch.argmax(output, dim=1)
            correct = pred == y.view(*pred.shape)
            #return loss.item(), torch.sum(correct).item(), len(x), y, pred
            losses = loss.item()
            corrects = torch.sum(correct).item()
            nums = len(x)

            test_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            test_accuracy = np.sum(corrects) / np.sum(nums) * 100
            ##F1 score calculate
            f1_score_mean = 0
            f1_average = 0
            test = True
            if test:
                for _x in range(len(x)):
                    img = x[_x].cpu().numpy()
                    img = img * 255
                    img = img.transpose(1,2,0)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    char_set = string.ascii_uppercase + string.digits
                    #fname= ''.join(random.sample(char_set*6, 6))

                    _pred = pred.cpu().numpy()
                    cv2.imwrite('./data/{}_{}.jpg'.format(_pred,fname[:-4]),img)
            # for i in range(len(pred)):
            #     _y_true = y[i].cpu().numpy()
            #     _y_pred = pred[i].cpu().numpy()
            #     f1_score_mean += f1_score(_y_true, _y_pred, average='weighted')
            # f1_average = f1_score_mean / len(pred)
        
            print(f"Test loss: {test_loss:.6f}\t"
              f"Test accruacy: {test_accuracy:.3f}%\t"
              f"F1 score:{f1_average:.3f}")


def convert_image_np(img):
    img = img.numpy().transpose((1, 2, 0)).squeeze()
    return img


def visualize_stn(dl, outfile):
    with torch.no_grad():
        data = next(iter(dl))[0]

        input_tensor = data.cpu()
        transformed_tensor = model.stn(data).cpu()

        input_grid = convert_image_np(make_grid(input_tensor))
        transformed_grid = convert_image_np(make_grid(transformed_tensor))

        # Plot the results side-by-side
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches((16, 16))
        ax[0].imshow(input_grid)
        ax[0].set_title('Dataset Images')
        ax[0].axis('off')

        ax[1].imshow(transformed_grid)
        ax[1].set_title('Transformed Images')
        ax[1].axis('off')

        plt.savefig(outfile)

def set_data_loader(args):
    test_data_set = CustomImageDataset(data_set_path=args.valid_path)
    test_loader = DataLoader(test_data_set, batch_size=1, shuffle=False)
    return test_loader

if __name__ == "__main__":
    # Evaluation settings
    parser = argparse.ArgumentParser(
        description='Traffic sign recognition evaluation script')
    parser.add_argument('--data', type=str, default='dataTL', metavar='D',
                        help="folder where data is located. test.p need to be found in the folder (default: data)")
    parser.add_argument('--model', type=str, default='trafficLight_model_64_32_best_rgb_0.956.pt', metavar='M',
                        help="the model file to be evaluated. (default: model.pt)")
    parser.add_argument('--outfile', type=str, default='visualize_stnTL.png', metavar='O',
                        help="visualize the STN transformation on some input batch (default: visualize_stn.png)")
    parser.add_argument('--valid-path', type=str, default='/dataset/test/', metavar='D',
                        help="Valid folder where data is located.")

    args = parser.parse_args()

    # Load model checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device)

    # Neural Network and Loss Function
    model = TrafficLightNet_64x32_noSTN(nclasses=7).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    ### Define train and test loader
    test_loader = set_data_loader(args)

    evaluate(model, criterion, test_loader)
    #visualize_stn(test_loader, args.outfile)