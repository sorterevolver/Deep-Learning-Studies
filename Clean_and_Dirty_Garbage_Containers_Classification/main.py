"""
These codes show how to test the deep learning model that has been successfully trained and saved.
"""
import os
import torch
from torchvision import datasets , transforms
from torch import nn
from torch.utils.data import DataLoader , random_split

"""
CDCM Dataset:
https://www.kaggle.com/datasets/rodrigolaguna/clean-dirty-containers-in-montevideo

After the CDCM dataset is downloaded from Kaggle, 
all clean labeled images should be put in a folder named "clean," 
and all dirty labeled images should be put in a folder named "dirty." 
Folders must be placed in the directory where the python file is.
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_ratio = 0.66

ROOT_PATH = "./clean-dirty/"

data_clean_dir = os.path.join(ROOT_PATH , 'clean')
data_dirty_dir = os.path.join(ROOT_PATH , 'dirty')

transform_global = transforms.Compose([
    transforms.ToTensor()
])

dataset_imgs = datasets.ImageFolder(os.path.join(ROOT_PATH) , transform=transform_global)

print(f"We have {len(dataset_imgs)} total Images with {len(dataset_imgs.classes)} classes")

train_set , test_set = random_split(dataset_imgs , [int(train_ratio * len(dataset_imgs)) ,
                                                    int(len(dataset_imgs) - int(
                                                        train_ratio * len(dataset_imgs)))])

# Resize, and the crop should be selected in the transform operation
# according to the size in which the model is most successful.
test_set.dataset.transform = transforms.Compose([
    transforms.Resize(512) ,
    transforms.CenterCrop(448) ,
    transforms.ToTensor()
])

class_categories = dataset_imgs.classes
num_classes = len(class_categories)

batch_size = 16

test_dataloader = torch.utils.data.DataLoader(dataset=test_set ,
                                              batch_size=batch_size)

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.11)
loss_fn = loss_fn.to(device)

#You need to download one of the most successful models from the 'model_download_address.txt' text file.
#After downloading, you must put the model file in the main directory.

#As an example:
model = torch.load('Vgg19_bn_size_448.pth')

# Network enters evaluation mode

test_loss_eval = []
test_accuracy_eval = []

correct_eval = 0

iterations_eval = 0

testing_loss_eval = 0.0

model.eval()
for i , (inputs , labels) in enumerate(test_dataloader):

    CUDA = torch.cuda.is_available()

    if CUDA:
        inputs_eval = inputs.cuda()
        labels_eval = labels.cuda()

    outputs_eval = model(inputs_eval)
    loss_eval = loss_fn(outputs_eval , labels_eval)
    testing_loss_eval += loss_eval.item()

    _ , predicted_eval = torch.max(outputs_eval , 1)
    correct_eval += (predicted_eval == labels_eval).sum()
    iterations_eval += 1

test_loss_eval.append(testing_loss_eval / iterations_eval)
test_accuracy_eval.append((100 * correct_eval / len(test_set)))
print(
    'Test_Loss: {:.3f}, Test_Accuracy: {:.3f} '
    .format(test_loss_eval[-1] ,
            test_accuracy_eval[-1]))
