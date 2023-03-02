# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
sys.path.append(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\transeffunet3d\research-contributions\UNETR\BTCV")
from typing import Tuple, Union

import torch
import torch.nn as nn

from MONAI.monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from MONAI.monai.networks.blocks.dynunet_block import UnetOutBlock
from MONAI.monai.networks.nets import ViT
from EfficientNet_PyTorch_3D.efficientnet_pytorch_3d.model import EfficientNet3D

class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")
        
        self.hybrid = EfficientNet3D.from_name("efficientnet-b7", in_channels=1).cuda()

        self.num_layers = 12
        # self.patch_size = (16, 16, 16)
        self.patch_size = (1, 1, 1)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        # print("feat_size:",self.feat_size) # (6, 6, 6)
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=2560,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=224,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=224,
            out_channels=80,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=80,
            out_channels=48,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=48,
            out_channels=32,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)  # type: ignore

    def proj_feat(self, x, hidden_size, feat_size):
        # print("before profect:",x.size(), feat_size)
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        # print("after project:",x.size())
        return x

    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights
            # copy weights from patch embedding
            for i in weights["state_dict"]:
                print(i)
            self.vit.patch_embedding.position_embeddings.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.position_embeddings_3d"]
            )
            self.vit.patch_embedding.cls_token.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.cls_token"]
            )
            self.vit.patch_embedding.patch_embeddings[1].weight.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.weight"]
            )
            self.vit.patch_embedding.patch_embeddings[1].bias.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.bias"]
            )

            # copy weights from  encoding blocks (default: num of blocks: 12)
            for bname, block in self.vit.blocks.named_children():
                print(block)
                block.loadFrom(weights, n_block=bname)
            # last norm layer of transformer
            self.vit.norm.weight.copy_(weights["state_dict"]["module.transformer.norm.weight"])
            self.vit.norm.bias.copy_(weights["state_dict"]["module.transformer.norm.bias"])

    def forward(self, x_in):
        original_resolution_skip = self.encoder1(x_in)
        x, features = self.hybrid.extract_features(x_in)

        # print("after hybrid:",x.size()) # 1,2560,3,3,3
        x,_ = self.vit(x)
        # print("after vit:",x.size()) # 27,768
        x = self.proj_feat(x, hidden_size=self.hidden_size, feat_size=(3,3,3))
        # print("after project:",x.size()) #  torch.Size([1, 768, 3, 3, 3])
        x = self.decoder5(x,features[3])
        # print("after decode5:",x.size()) # torch.Size([1, 224, 6, 6, 6])
        x = self.decoder4(x, features[2])
        # print("after decode 4:",x.size()) # torch.Size([1, 80, 12, 12, 12])
        x = self.decoder3(x, features[1])
        # print("after decode 3:",x.size()) #torch.Size([1, 48, 24, 24, 24])
        x = self.decoder2(x, features[0])
        # print("after decode 2:",x.size()) # torch.Size([1, 32, 48, 48, 48])
        x = self.decoder1(x, original_resolution_skip)
        # print("after decode 1:",x.size()) # torch.Size([1, 16, 96, 96, 96])
        logits = self.out(x)
        return logits
if __name__ == "__main__":
    device = torch.device("cuda")
    # input = torch.randn((1, 2560, 3, 3, 3)).to(device)
    input = torch.randn((1, 1, 96, 96, 96)).to(device)
    model = UNETR(
    in_channels=1,
    out_channels=14,
    img_size=(96,96,96),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    conv_block=True,
    res_block=True,
    dropout_rate=0.0,
).to(device)

    model.train()
    output = model(input)
    print(output.size())
