import numpy as np
from osgeo import gdal
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from glob import glob
import torch.optim as optim
from torch import nn
from tqdm import tqdm


def read_image(file_path):
    dataset = gdal.Open(file_path)
    if dataset is None:
        raise FileNotFoundError(f"Image {file_path} not found.")
    band = dataset.GetRasterBand(1)
    image = band.ReadAsArray()
    return image


def split_image(image, patch_size=33):
    patches = []
    h, w = image.shape
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            if patch.shape == (patch_size, patch_size):
                patches.append(patch)
    return patches


def downsample(image, scale=3):
    return cv2.resize(image, (image.shape[1] // scale, image.shape[0] // scale), interpolation=cv2.INTER_CUBIC)


def compute_gradient(image):
    # 确保输入图像是2D的
    if len(image.shape) == 3:
        image = image[0, :, :]  # 取第一个通道


    # 计算梯度的函数，使用Sobel算子
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    return gradient


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

        # 将低分辨率图像放大到高分辨率图像的大小
        low_res_upscaled = torch.nn.functional.interpolate(low_res_tensor.unsqueeze(0), size=(33, 33), mode='bicubic',
                                                           align_corners=False).squeeze(0)

        input_tensor = torch.cat((low_res_upscaled, high_res_gradient_tensor), dim=0)
        return input_tensor, high_res_tensor


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


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        batch_size = output.size(0)
        gradient_output = torch.zeros_like(output)
        gradient_target = torch.zeros_like(target)

        for i in range(batch_size):
            gradient_output[i] = torch.from_numpy(compute_gradient(output[i].detach().cpu().numpy())).float().cuda()
            gradient_target[i] = torch.from_numpy(compute_gradient(target[i].detach().cpu().numpy())).float().cuda()

        return self.mse(output, target) + self.mse(gradient_output, gradient_target)


# 数据加载
image_paths = glob('C:\\Users\\Lenovo\\Desktop\\DEM\\cropped_dted.tif')  # 修改为你数据的路径
dataset = CustomDataset(image_paths)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 初始化模型和优化器
model = SRCNN(num_channels=2).cuda()
criterion = GradientLoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练循环
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for inputs, targets in tqdm(dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}')
torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')
