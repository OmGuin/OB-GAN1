import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# los dependencies sus

import torch
import torchvision
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.utils as vutils
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time


class GeneratorLung(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(GeneratorLung, self).__init__()
        self.net = nn.Sequential(
            self._block(z_dim, features_g * 32, 7, 1, 0),
            self._block(features_g * 32, features_g * 16, 3, 2, 1),
            self._block(features_g * 16, features_g * 8, 3, 2, 1),
            self._block(features_g * 8, features_g * 4, 3, 2, 1),
            self._block(features_g * 4, features_g * 2, 2, 2, 1),
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1, ),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class GeneratorNodule(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(GeneratorNodule, self).__init__()
        self.net = nn.Sequential(
            self._block(z_dim, features_g * 2, 2, 1, 0),
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=2, stride=2, padding=1, ),  # img:54x54
            nn.Sigmoid(),

        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        # load Faster RCNN pre-trained model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # get the number of input features
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # define a new head for the detector with required number of classes
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets):
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
                boxes: float32 - torch.Tensor([[xmin1, ymin1, xmax1, ymax1], [xmin2, ymin2, xmax2, ymax2], ...])
                area = area of boxes
                labels = labels of boxes
                iscrowd
                image_id
        """

        loss_dict = self.model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        return loss_value


class GAN(nn.Module):
    def __init__(self, num_classes):
        super(GAN, self).__init__()

        self.generatorlung = GeneratorLung(z_dim=27, channels_img=1, features_g=32)
        self.generatornodule = GeneratorNodule(z_dim=27, channels_img=1, features_g=32)
        self.discriminator = Discriminator(num_classes=num_classes)

    def forward_one(self, image, target):
        """
        Args:
            images (Tensor): image to be processed
            targets (Dict[str, Tensor]): ground-truth boxes present in the image (optional)
        """

        noise = torch.randn(1, 27, 27, 27)

        generated_lungimage = self.generatorlung(noise)[0]
        generated_nodule = self.generatornodule(noise)[0]

        # Image doesnt need autograd
        image.requires_grad = False

        generatorlung_loss = F.binary_cross_entropy(generated_lungimage, image)

        avg_nodule = torch.full((1, 54, 54), 0.6)

        # Might be scuffed
        generatornodule_loss = F.binary_cross_entropy(avg_nodule, generated_nodule)

        # This is to allow autograd to do autograd of generated lung image without interference and not mess up the computation graph
        generated_lungimage_with_nodules = generated_lungimage.clone().detach()

        # Replace the pixels in the bounding box with the nodule
        for bbox in target["boxes"]:
            xmin = int(bbox[0].item())
            ymin = int(bbox[1].item())
            xmax = int(bbox[2].item())
            ymax = int(bbox[3].item())

            resize_transform = torchvision.transforms.Resize(size=(ymax-ymin, xmax-xmin))

            resized_generated_nodule = resize_transform(generated_nodule)

            generated_lungimage_with_nodules[:, ymin:ymax, xmin:xmax] = resized_generated_nodule


        loss_value_bbox_realimage_discriminator = self.discriminator(images=[image], targets=[target])
        loss_value_bbox_generatedimage_discriminator = self.discriminator(images=[generated_lungimage_with_nodules], targets=[target])

        discriminator_loss = loss_value_bbox_realimage_discriminator - (generatorlung_loss + generatornodule_loss) + loss_value_bbox_generatedimage_discriminator

        return [generatorlung_loss, generatornodule_loss, discriminator_loss]

    def forward(self, images, targets):
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the images (optional)
        """

        all_losses = [self.forward_one(images[i], targets[i]) for i in range(len(images))]
        sum_generatorlung_loss = sum([all_losses[i][0] for i in range(len(all_losses))])
        sum_generatornodule_loss = sum([all_losses[i][1] for i in range(len(all_losses))])
        sum_discriminator_loss = sum([all_losses[i][2] for i in range(len(all_losses))])

        return [sum_generatorlung_loss, sum_generatornodule_loss, sum_discriminator_loss]


from config import (
    DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR,
    VISUALIZE_TRANSFORMED_IMAGES, NUM_WORKERS,
)
from custom_utils import Averager, SaveBestModel, save_model, save_loss_plot
from tqdm.auto import tqdm
from datasets import (
    create_train_dataset, create_valid_dataset,
    create_train_loader, create_valid_loader
)
import torch

import matplotlib.pyplot as plt

plt.style.use('ggplot')


# function for running training iterations
def train(train_data_loader, model):
    print('Training')
    global train_itr
    global train_loss_list

    # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))

    for i, data in enumerate(prog_bar):
        optimizer_genlung.zero_grad()
        optimizer_gennodule.zero_grad()
        optimizer_disc.zero_grad()

        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        generatorlung_loss, generatornodule_loss, discriminator_loss = model(images, targets)

        generatorlung_loss_value = generatorlung_loss.item()
        generatorlung_loss_list.append(generatorlung_loss_value)
        generatorlung_loss_hist.send(generatorlung_loss_value)

        generatornodule_loss_value = generatornodule_loss.item()
        generatornodule_loss_list.append(generatornodule_loss_value)
        generatornodule_loss_hist.send(generatornodule_loss_value)

        discriminator_loss_value = discriminator_loss.item()
        discriminator_loss_list.append(discriminator_loss_value)
        discriminator_loss_hist.send(discriminator_loss_value)

        generatorlung_loss.backward()
        generatornodule_loss.backward()
        discriminator_loss.backward()

        optimizer_genlung.step()
        optimizer_gennodule.step()
        optimizer_disc.step()

        train_itr += 1

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(
            desc=f"Loss for Lung Generator: {generatorlung_loss_value:.4f}. Loss for Nodule Generator: {generatornodule_loss_value:.4f}. Loss for Discriminator: {discriminator_loss_value:.4f}.")

    return [generatorlung_loss_list, generatornodule_loss_list, discriminator_loss_list]


# function for running validation iterations
def validate(valid_data_loader, model):
    print('Validating')
    global val_itr
    global val_loss_list

    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

    for i, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            generatorlung_loss, generatornodule_loss, discriminator_loss = model(images, targets)

        generatorlung_loss_value = generatorlung_loss.item()
        val_generatorlung_loss_list.append(generatorlung_loss_value)
        val_generatorlung_loss_hist.send(generatorlung_loss_value)

        generatornodule_loss_value = generatornodule_loss.item()
        val_generatornodule_loss_list.append(generatornodule_loss_value)
        val_generatornodule_loss_hist.send(generatornodule_loss_value)

        discriminator_loss_value = discriminator_loss.item()
        val_discriminator_loss_list.append(discriminator_loss_value)
        val_discriminator_loss_hist.send(discriminator_loss_value)

        val_itr += 1

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(
            desc=f"Loss for Lung Generator: {generatorlung_loss_value:.4f}. Loss for Nodule Generator: {generatornodule_loss_value:.4f}. Loss for Discriminator: {discriminator_loss_value:.4f}.")

    return [val_generatorlung_loss_list, val_generatornodule_loss_list, val_discriminator_loss_list]


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    train_dataset = create_train_dataset()
    valid_dataset = create_valid_dataset()
    train_loader = create_train_loader(train_dataset, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    # initialize the model and move to the computation device
    model = GAN(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    # get the model parameters
    params_genlung = [p for p in model.generatorlung.parameters() if p.requires_grad]
    optimizer_genlung = torch.optim.SGD(params_genlung, lr=0.001, momentum=0.9, weight_decay=0.0005)

    params_gennodule = [p for p in model.generatornodule.parameters() if p.requires_grad]
    optimizer_gennodule = torch.optim.SGD(params_gennodule, lr=0.001, momentum=0.9, weight_decay=0.0005)

    params_disc = [p for p in model.discriminator.parameters() if p.requires_grad]
    optimizer_disc = torch.optim.SGD(params_disc, lr=0.001, momentum=0.9, weight_decay=0.0005)

    # initialize the Averager class
    generatorlung_loss_hist = Averager()
    generatornodule_loss_hist = Averager()
    discriminator_loss_hist = Averager()

    val_generatorlung_loss_hist = Averager()
    val_generatornodule_loss_hist = Averager()
    val_discriminator_loss_hist = Averager()

    train_itr = 1
    val_itr = 1
    # train and validation loss lists to store loss values of all...
    # ... iterations till ena and plot graphs for all iterations
    generatorlung_loss_list = []
    generatornodule_loss_list = []
    discriminator_loss_list = []

    val_generatorlung_loss_list = []
    val_generatornodule_loss_list = []
    val_discriminator_loss_list = []

    # name to save the trained model with
    MODEL_NAME = 'scl-DC7G3GANN'

    # whether to show transformed images from data loader or not
    if VISUALIZE_TRANSFORMED_IMAGES:
        from custom_utils import show_tranformed_image

        show_tranformed_image(train_loader)

    # initialize SaveBestModel class
    save_best_model = SaveBestModel()

    # start the training epochs
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch + 1} of {NUM_EPOCHS}")

        # reset the training and validation loss histories for the current epoch
        generatorlung_loss_hist.reset()
        generatornodule_loss_hist.reset()
        discriminator_loss_hist.reset()

        val_generatorlung_loss_hist.reset()
        val_generatornodule_loss_hist.reset()
        val_discriminator_loss_hist.reset()

        # start timer and carry out training and validation
        start = time.time()
        generatorlung_loss_list, generatornodule_loss_list, discriminator_loss_list = train(train_loader, model)
        val_generatorlung_loss_list, val_generatornodule_loss_list, val_discriminator_loss_list = validate(valid_loader,
                                                                                                           model)

        print(
            f"Epoch #{epoch + 1}. Train. Loss for Lung Generator: {generatorlung_loss_hist:.4f}. Loss for Nodule Generator: {generatornodule_loss_hist:.4f}. Loss for Discriminator: {discriminator_loss_hist:.4f}.")
        print(
            f"Epoch #{epoch + 1}. Validation. Loss for Lung Generator: {val_generatorlung_loss_hist:.4f}. Loss for Nodule Generator: {val_generatornodule_loss_hist:.4f}. Loss for Discriminator: {val_discriminator_loss_hist:.4f}.")

        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        # save loss plot
        save_loss_plot(OUT_DIR, generatorlung_loss_list, val_generatorlung_loss_list)
        save_loss_plot(OUT_DIR, generatornodule_loss_list, val_generatornodule_loss_list)
        save_loss_plot(OUT_DIR, discriminator_loss_list, val_discriminator_loss_list)
