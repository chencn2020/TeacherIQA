import torch
import torchvision
from PIL import Image
import numpy as np
from model import studentNetwork as IQAModel
import argparse
import warnings
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
warnings.filterwarnings("ignore")

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def predict_IQA_Score(config):
    im_path = config.input_image

    # load the model
    model_hyper = IQAModel.StudentNetwork().cuda()
    model_hyper.load_state_dict((torch.load(config.pre_train_model)))
    model_hyper.eval()

    # define the way of transforming.
    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.RandomCrop(size=224),
                        torchvision.transforms.ToTensor()])

    img_or = pil_loader(im_path)
    pred_scores = []

    # crop the image 25 times
    for time in range(config.crop_times):
        img = transforms(img_or)
        img = torch.tensor(img.cuda()).unsqueeze(0)
        pred = model_hyper(img)
        pred_scores.append(float(pred.item()))

    # calculate the average score.
    score = np.mean(pred_scores)

    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', dest='input_image', type=str, required=True)
    parser.add_argument('--pre_train_model', dest='pre_train_model',  type=str, required=True)
    parser.add_argument('--crop_times', dest='crop_times', type=int, default=25)
    config = parser.parse_args()

    score = predict_IQA_Score(config)
    print('Final Average Predicted Quality Score: {}'.format(round(score, 2)))
