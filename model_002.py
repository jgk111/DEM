import numpy as np
from osgeo import gdal
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from glob import glob
import torch.optim as optim
from torch import nn
from tqdm import tqdm
import torchvision

# 读取图像
def read_image(file_path):
    dataset = gdal.Open(file_path)
    if dataset is None:
        raise FileNotFoundError(f"Image {file_path} not found.")
    band = dataset.GetRasterBand(1)
    image = band.ReadAsArray()
    return image

# 切分图像
def split_image(image, patch_size=33):
    patches = []
    h, w = image.shape
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            if patch.shape == (patch_size, patch_size):
                patches.append(patch)
    return patches

# 降采样
def downsample(image, scale=3):
    return cv2.resize(image, (image.shape[1] // scale, image.shape[0] // scale), interpolation=cv2.INTER_CUBIC)

# 计算梯度
def compute_gradient(image):
    if len(image.shape) == 3:
        image = image[0, :, :]  # 取第一个通道
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    return gradient

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, image_paths, patch_size=33, scale=3):
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.scale = scale
        self.patches = self.load_patches()
        if len(self.patches) == 0:
            raise ValueError("No patches loaded. Check your image paths and patch size.")
        print(f"Loaded {len(self.patches)} patches from {len(self.image_paths)} images.")

    def load_patches(self):
        patches = []
        for image_path in self.image_paths:
            print(f"Reading image: {image_path}")
            image = read_image(image_path)
            image_patches = split_image(image, self.patch_size)
            for patch in image_patches:
                low_res_patch = downsample(patch, self.scale)
                patches.append((patch, low_res_patch))
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        high_res, low_res = self.patches[idx]
        high_res_gradient = compute_gradient(high_res)

        high_res_tensor = torch.from_numpy(high_res).float().unsqueeze(0) / 255.0
        high_res_gradient_tensor = torch.from_numpy(high_res_gradient).float().unsqueeze(0) / 255.0
        low_res_tensor = torch.from_numpy(low_res).float().unsqueeze(0) / 255.0

        low_res_upscaled = torch.nn.functional.interpolate(low_res_tensor.unsqueeze(0), size=(33, 33), mode='bicubic', align_corners=False).squeeze(0)

        input_tensor = torch.cat((low_res_upscaled, high_res_gradient_tensor), dim=0)
        return input_tensor, high_res_tensor

# SRCNN模型
class SRCNN(nn.Module):
    def __init__(self, num_channels=2):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# 感知损失
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features[:18].eval().cuda()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        output_3ch = torch.cat([output] * 3, dim=1)
        target_3ch = torch.cat([target] * 3, dim=1)
        output_features = self.vgg(output_3ch)
        target_features = self.vgg(target_3ch)
        perceptual_loss = self.mse(output_features, target_features)
        return perceptual_loss

# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 混合损失
class HybridLoss(nn.Module):
    def __init__(self, discriminator):
        super(HybridLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss().cuda()
        self.adversarial_loss = nn.BCEWithLogitsLoss().cuda()
        self.discriminator = discriminator

    def forward(self, output, target):
        mse_loss = self.mse(output, target)
        perceptual_loss = self.perceptual_loss(output, target)

        real_labels = torch.ones_like(self.discriminator(target))
        fake_labels = torch.zeros_like(self.discriminator(output))

        real_loss = self.adversarial_loss(self.discriminator(target), real_labels)
        fake_loss = self.adversarial_loss(self.discriminator(output), fake_labels)

        adversarial_loss = (real_loss + fake_loss) / 2

        return mse_loss + perceptual_loss + adversarial_loss

# 数据加载
image_paths = glob('C:\\Users\\Lenovo\\Desktop\\DEM\\cropped_dted.tif')  # 修改为你数据的路径
dataset = CustomDataset(image_paths)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 初始化模型和优化器
model = SRCNN(num_channels=2).cuda()
discriminator = Discriminator().cuda()
criterion = HybridLoss(discriminator).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)

# 训练循环
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    discriminator.train()
    epoch_loss = 0
    for inputs, targets in tqdm(dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()

        # 训练判别器
        discriminator_optimizer.zero_grad()
        real_outputs = discriminator(targets)
        fake_outputs = discriminator(model(inputs).detach())
        real_loss = criterion.adversarial_loss(real_outputs, torch.ones_like(real_outputs))
        fake_loss = criterion.adversarial_loss(fake_outputs, torch.zeros_like(fake_outputs))
        discriminator_loss = (real_loss + fake_loss) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # 训练生成器
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}')

# 保存模型
torch.save(model.state_dict(), f'model_epoch_{ 1}.pth')
