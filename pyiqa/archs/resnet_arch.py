import torch.nn as nn
import timm
from pyiqa.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class Resnet(nn.Module):

    def __init__(self, model_name="resnet18", pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    model = Resnet(model_name="resnet18")

    print("done!")
