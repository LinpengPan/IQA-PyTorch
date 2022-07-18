import torch
import torch.nn as nn
import timm
from pyiqa.utils.registry import ARCH_REGISTRY
import torch.nn.functional as F
from torchvision.models import resnet50


@ARCH_REGISTRY.register()
class BaseCNN(nn.Module):

    def __init__(self, base_model_name="resnet18", d_model=512):
        super(BaseCNN, self).__init__()
        self.base_model = timm.create_model(model_name=base_model_name,
                                            pretrained=True, features_only=True,
                                            out_indices=[1, 2, 3, 4])
        self.num_channels_encoder = 960 if base_model_name in ("resnet18", "resnet34") else 3840
        self.num_channels_decoder = 512 if base_model_name in ("resnet18", "resnet34") else 2048
        self.conv = nn.Conv2d(self.num_channels_encoder, d_model, 1)
        self.fc = nn.Linear(7 * 7 * d_model, 1)  # 目前该模型只能处理224×224的输入

    def forward_patch(self, x):
        """
        x的尺寸尽量不要使得下采样的过程中出现无法整除的情况
        """
        base_feats = self.base_model(x)  # 池化时向上取整
        x = base_feats[0]
        for i in range(3):
            x = F.avg_pool2d(x, 2)  # 向下取整
            diffY = base_feats[i + 1].size(2) - x.size(2)
            diffX = base_feats[i + 1].size(3) - x.size(3)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
            x = torch.concat([x, base_feats[i + 1]], dim=1)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def random_crop_test(self, x, sample_num=25):
        b, c, h, w = x.shape
        th = tw = 224
        cropped_x = []
        for s in range(sample_num):
            i = torch.randint(0, h - th + 1, size=(1,)).item()
            j = torch.randint(0, w - tw + 1, size=(1,)).item()
            cropped_x.append(x[:, :, i:i + th, j:j + tw])
        cropped_x = torch.cat(cropped_x, dim=0)
        results = self.forward_patch(cropped_x)
        results = results.reshape(sample_num, b).mean(dim=0)
        return results.unsqueeze(-1)

    def forward(self, x):
        r"""
        Args:
            memory_mask:
            x: A distortion tensor. Shape :math:`(N, C, H, W)`.
        """
        # imagenet normalization of input is hard coded

        if self.training:
            return self.forward_patch(x)
        else:
            return self.random_crop_test(x)


if __name__ == '__main__':
    basecnn = BaseCNN()
    image = torch.randn((2, 3, 224, 224))
    output = basecnn(image)

    print("done!")
