import torch
import torch.nn as nn
import timm
from pyiqa.utils.registry import ARCH_REGISTRY
import torch.nn.functional as F


class BaseCNN(nn.Module):

    def __init__(self, base_model_name="resnet50"):
        super(BaseCNN, self).__init__()
        self.base_model = timm.create_model(model_name=base_model_name,
                                            pretrained=True, features_only=True,
                                            out_indices=[1, 2, 3, 4])
        self.num_channels_encoder = 960 if base_model_name in ("resnet18", "resnet34") else 3840
        self.num_channels_decoder = 512 if base_model_name in ("resnet18", "resnet34") else 2048

    def forward(self, x):
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
        return x, base_feats[3]


class AddPositionEmbs(nn.Module):

    def __init__(self, num_lengths, dim):
        super(AddPositionEmbs, self).__init__()
        self.position_emb = nn.parameter.Parameter(torch.randn(num_lengths, dim))
        nn.init.normal_(self.position_emb, std=0.02)

    def forward(self, inputs, inputs_position):

        return inputs + self.position_emb[:inputs_position, :].squeeze(0)


# 在该文件中debug的时候不能添加该注册器
@ARCH_REGISTRY.register()
class Caiqa(nn.Module):

    def __init__(self, base_model_name="resnet50", position=True, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation="relu",
                 custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-05,
                 norm_first=False, bias=True):
        super(Caiqa, self).__init__()
        self.basemodel = BaseCNN(base_model_name)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                          dim_feedforward, dropout, activation, custom_encoder,
                                          custom_decoder, layer_norm_eps, True, norm_first)
        self.encoder_proj = nn.Conv2d(self.basemodel.num_channels_encoder, d_model, kernel_size=1)
        self.decoder_proj = nn.Conv2d(self.basemodel.num_channels_decoder, d_model, kernel_size=1)

        self.quality = nn.parameter.Parameter(torch.zeros(1, 1, d_model))  # 可学习的质量嵌入
        self.num_length = 50
        self.position = position
        if position:
            self.position = AddPositionEmbs(self.num_length, d_model)
        self.head = nn.Linear(d_model, 1, bias)

    def forward_patch(self, x, src_mask=None, tgt_mask=None, memory_mask=None):  # 后续需要看是否真的需要用到mask，需要的话需要重写model了。
        """
        目前的位置编码只能适用于固定长度，需要考虑如何解决任意长度的位置编码,因此算法目前只能接受224 乘 224 大小的输入。
        """
        assert x.shape[2:] == torch.Size([224, 224]), f'Input patch size must be (224, 224), but got {x.shape[2:]}'
        encoder_input, decoder_input = self.basemodel(x)
        encoder_input = self.encoder_proj(encoder_input)
        decoder_input = self.decoder_proj(decoder_input)

        encoder_input = torch.reshape(encoder_input, [encoder_input.size(0), -1, encoder_input.size(1)])  # B N C
        decoder_input = torch.reshape(decoder_input, [decoder_input.size(0), -1, decoder_input.size(1)])
        if self.position:
            encoder_input = self.position(encoder_input, encoder_input.size(1))
        quality_token = self.quality.repeat(decoder_input.size(0), 1, 1)
        decoder_input = torch.cat([quality_token, decoder_input], dim=1)
        if self.position:
            decoder_input = self.position(decoder_input, decoder_input.size(1))
        out = self.transformer(encoder_input, decoder_input, src_mask, tgt_mask, memory_mask)
        return self.head(out[:, 0, :])

    def random_crop_test(self, x, sample_num=25, src_mask=None, tgt_mask=None, memory_mask=None):
        b, c, h, w = x.shape
        th = tw = 224
        cropped_x = []
        for s in range(sample_num):
            i = torch.randint(0, h - th + 1, size=(1,)).item()
            j = torch.randint(0, w - tw + 1, size=(1,)).item()
            cropped_x.append(x[:, :, i:i + th, j:j + tw])
        cropped_x = torch.cat(cropped_x, dim=0)
        results = self.forward_patch(cropped_x, src_mask, tgt_mask, memory_mask)
        results = results.reshape(sample_num, b).mean(dim=0)
        return results.unsqueeze(-1)

    def forward(self, x, src_mask=None, tgt_mask=None, memory_mask=None):
        r"""
        Args:
            memory_mask:
            x: A distortion tensor. Shape :math:`(N, C, H, W)`.
        """
        # imagenet normalization of input is hard coded

        if self.training:
            return self.forward_patch(x, src_mask, tgt_mask, memory_mask)
        else:
            return self.random_crop_test(x, src_mask=src_mask,
                                         tgt_mask=tgt_mask, memory_mask=memory_mask)


if __name__ == "__main__":
    # """
    # features_only支持把模型转换成特征提取器，同时也支持使用out_indices=[...]来指定需要输出哪几层的特征图。
    # resnet系例默认输出conv1到conv5的输出，它这里的输入支持任意尺寸，正常情况下，使用的都是conv2到conv5的输出。
    # """
    # resnet50 = timm.create_model(model_name="resnet50",
    #                              pretrained=True, features_only=True)
    # image = torch.randn((1, 3, 300, 310))
    # output = resnet50(image)  # output是resnet50
    #
    # print("done")

    # basecnn = BaseCNN()
    # image = torch.randn((2, 3, 300, 500))
    # output = basecnn(image)
    #
    # print("done!")
    caiqa = Caiqa(base_model_name="resnet18")
    image = torch.randn((96, 3, 224, 224))
    output = caiqa(image)

    print("done!")
