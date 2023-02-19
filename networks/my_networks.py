import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("E:/tai_lieu_hoc_tap/tdh/tuannca_datn")
# transformer block
from aiplatform.TransUNet.networks.vit_seg_modeling import Block
from aiplatform.pvtv2.pvtv2 import Block


from aiplatform.TransUNet.networks.vit_seg_modeling_resnet_skip import ResNetV2
from aiplatform.pvtv2.pvtv2 import PyramidVisionTransformerV2 as PVT
from aiplatform.pvtv2.pvtv2 import OverlapPatchEmbed
# from aiplatform.Swin_Unet.networks.swin_transformer_unet_skip_expand_decoder_sys import 

class DownDimension(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(DownDimension,self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.down = nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.down(x)
        return x


class PatchExpand(nn.Module):
    def __init__(self,ch_in,ch_out, final=False):
        super(PatchExpand,self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.final = final
        # self.up = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		#     nn.BatchNorm2d(ch_out),
		# 	nn.ReLU(inplace=True)
        # )
        # self.upX2 = nn.Upsample(scale_factor=2)
        if final:
            self.upX2 =  nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.upX2 =  nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(self.ch_in,self.ch_out,kernel_size=3,stride=1,padding=1,bias=True)
        self.bn = nn.BatchNorm2d(self.ch_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        # x = self.up(x)
        x = self.upX2(x)
        # print("after upsample:",x.size())
        x = self.conv(x)
        # print("after conv:",x.size())
        x = self.bn(x)
        x = self.relu(x)
        # print("after relu:",x.size())
        _,_,H,W = x.shape
        if not self.final:
            x = x.flatten(2).transpose(1, 2)
        # print("after flatten:",x.size())
        return x,H,W

class MyBlock(nn.Module):
    def __init__(self, config, in_ch, patch_size, stride, padding, dim):
        super(MyBlock, self).__init__()
        self.transformer = Block(config=config, vis=False)
        n_patches = 256
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = nn.Dropout(0.1)
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.patch_merging = nn.Conv2d()

    def forward(self, x):
        # input: x.size() = C,H,W
        x = x.flatten(2).transpose(-1,-2) # (256,768)
        embeddings = x + self.position_embeddings # (256,768)
        embeddings = self.dropout(embeddings) # (256,768)
        # transformer
        hidden_states, weights = self.transformer(embeddings) 
        encoded = self.encoder_norm(hidden_states) # (266,768)
        encoded = self.patch_merging(encoded) # reshape from (n_patches,dims) to (C,H,W)
        return encoded

class MyEncoder(nn.Module):
    def __init__(self):
        super(MyEncoder, self).__init__()
        self.hybrid_model = ResNetV2(block_units=(3, 4, 9), width_factor=1) # follow transunet r50_b16
        self.patch_embeddings = nn.Conv2d(in_channels=1024, 
                                         out_channels=768,
                                         kernel_size=(1,1),
                                         stride=(1,1))
        self.position_embeddings = nn.Parameter(torch.zeros(1, 256, 768))
        self.dropout = nn.Dropout(0.1) # dropout_rate = 0.1
        self.block1 = MyBlock()
        self.block2 = MyBlock()
        self.block3 = MyBlock()
        self.block4 = MyBlock()
        self.features = []
        
    def forward(self,x):
        features = []
        x, _ = self.hybrid_model(x) # (1024,16,16)
        x = self.patch_embeddings(x) # (768,16,16)

        for i in range(4):
            x = self.block(x) # (dims, H, W)
            features.append(x)
        return self.features


class MyDecoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False):
        super(MyDecoder, self).__init__()

        self.num_stages = num_stages
        
        self.patch_expand1 = PatchExpand(ch_in=512, ch_out=256)
        self.patch_expand2 = PatchExpand(ch_in=256, ch_out=128)
        self.patch_expand3 = PatchExpand(ch_in=128, ch_out=64)
        self.patch_expand4 = PatchExpand(ch_in=64, ch_out=32)
        self.patch_expand_final = PatchExpand(ch_in=32, ch_out=32, final=True)

        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[-2],
            num_heads=num_heads[-2],
            mlp_ratio=4,
            qkv_bias=False,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            sr_ratio=sr_ratios[-1],
            linear=False
        ) for j in range(2)])

        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[-3],
            num_heads=num_heads[-3],
            mlp_ratio=4,
            qkv_bias=False,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            sr_ratio=sr_ratios[-3],
            linear=False
        ) for j in range(2)])

        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[-4],
            num_heads=num_heads[-4],
            mlp_ratio=4,
            qkv_bias=False,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            sr_ratio=sr_ratios[-4],
            linear=False
        ) for j in range(2)])

        self.block4 = nn.ModuleList([Block(
            dim=32,
            num_heads=1,
            mlp_ratio=4,
            qkv_bias=False,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            sr_ratio=sr_ratios[-4],
            linear=False
        ) for j in range(2)])

        self.down_dim_512_256 = DownDimension(ch_in=512, ch_out=256)
        self.down_dim_256_128 = DownDimension(ch_in=256, ch_out=128)
        self.down_dim_128_64 = DownDimension(ch_in=128, ch_out=64)

    def forward(self, features):
        B=features[0].shape[0]
        # print(self.block1)
        # stage 1
        # print("stage1: before expand x=",x.size()) # torch.Size([2, 512, 7, 7])
        x,H,W = self.patch_expand1(features[-1])
        # print("stage1: after expand x=",x.size()) # torch.Size([2, 49, 256])
        for blk in self.block1:
            x=blk(x,H,W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() 
        # print("stage1: after block x=",x.size()) # torch.Size([2, 256, 14, 14])
        x = torch.cat([x, features[-2]], dim=1) # torch.Size([2, 512, 14, 14])
        x = self.down_dim_512_256(x)
        # print("after down dimension:",x.size()) # torch.Size([2, 256, 14, 14])

        # stage 2
        x,H,W = self.patch_expand2(x)
        for blk in self.block2:
            x=blk(x,H,W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = torch.cat([x, features[-3]], dim=1)
        x = self.down_dim_256_128(x)
        # print("after down dimension:",x.size()) # torch.Size([2, 128, 28, 28])
        
        # stage 3
        x,H,W = self.patch_expand3(x)
        for blk in self.block3:
            x=blk(x,H,W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = torch.cat([x, features[-4]], dim=1) # 128,56,56
        x = self.down_dim_128_64(x) # 64,56,56

        # stage 4
        x, H,W = self.patch_expand4(x)
        for blk in self.block4:
            x=blk(x,H,W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        x,H,W = self.patch_expand_final(x)
        # print("final:",x.size()) # torch.Size([2, 32, 224, 224])
        return x

class SegmentationHead(nn.Module):
    def __init__(self, num_classes=9):
        super(SegmentationHead, self).__init__()
        self.up_dimention = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(32),
			nn.ReLU(inplace=True)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(64),
			nn.ReLU(inplace=True)
        )
        self.linear = nn.Conv2d(64, num_classes, kernel_size=3, padding=1, bias=True)
    def forward(self, skip, x):
        skip = self.up_dimention(skip)
        x = torch.cat([x,skip], dim=1)
        x = self.conv(x)
        x = self.conv(x)
        x = self.linear(x)
        # print(x.size())
        return x


import matplotlib.pyplot as plt
class MyNetworks(nn.Module):
    def __init__(self, img_size=224, num_classes=9):
        super(MyNetworks, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size

        # self.encoder = MyEncoder() # 
        self.encoder = PVT(img_size=self.img_size)
        self.decoder = MyDecoder()
        self.head = SegmentationHead()

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        skip = x.clone()
        features = self.encoder(x)
        # x = self.decoder(x, features)
        # check segmentation_head sau!!!
        # print("my_networks: len(features): ", len(features)) # 4
        # print(features[0].size(), features[1].size(), features[2].size(), features[3].size())
        # print("features[-1]:",features[-1].size()) # torch.Size([2, 512, 7, 7])
        x = self.decoder(features)
        # print("after decoder:",x.size()) # torch.Size([2, 32, 224, 224])
        x = self.head(skip,x)
        # print(x.size())
        # # torch.Size([2, 64, 56, 56]) torch.Size([2, 128, 28, 28]) torch.Size([2, 256, 14, 14]) torch.Size([2, 512, 7, 7])
        return x

if __name__ == "__main__":
    # patch_expand = PatchExpand(ch_in=512, ch_out=256)
    # x = torch.randn(1, 512, 7, 7)
    # batch = x.shape[0]
    # x,H,W = patch_expand(x) # 3136,64
    # x = x.reshape(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()
    # print(x.size())

    x = torch.randn(2, 3, 224, 224)
    net = MyNetworks()
    x=net(x)