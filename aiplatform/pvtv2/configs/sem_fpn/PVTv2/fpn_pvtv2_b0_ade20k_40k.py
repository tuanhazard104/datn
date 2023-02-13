_base_ = './fpn_pvtv2_b2_ade20k_40k.py'

# model settings
model = dict(
    pretrained='pretrained/pvt_v2_b0.pth',
    backbone=dict(
        type='pvt_v2_b0',
        style='pytorch'),
    neck=dict(in_channels=[32, 64, 160, 256]))
