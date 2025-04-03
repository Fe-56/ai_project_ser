import torch.nn.functional as F
import torchvision.transforms as T
import torch


class AveragePoolResize:
    def __init__(self, output_size=(224, 224)):
        self.output_size = output_size  # (H, W)

    def __call__(self, img_tensor):
        """
        img_tensor: Tensor of shape [C, H, W] with pixel values in [0, 1]
        """
        _, H, W = img_tensor.shape
        kernel_size = (H // self.output_size[0], W // self.output_size[1])
        return F.avg_pool2d(img_tensor.unsqueeze(0), kernel_size=kernel_size, stride=kernel_size).squeeze(0)
