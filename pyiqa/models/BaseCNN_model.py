import torch

from pyiqa.utils.registry import MODEL_REGISTRY
from .general_iqa_model import GeneralIQAModel


@MODEL_REGISTRY.register()
class BaseCNNModel(GeneralIQAModel):

    def test(self):  # 这里的test可以用来对图像进行切块测试。
        self.net.eval()
        with torch.no_grad():
            self.output_score = self.get_bare_model(self.net).random_crop_test(self.img_input)
        self.net.train()