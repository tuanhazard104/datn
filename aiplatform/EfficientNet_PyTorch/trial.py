from efficientnet_pytorch import EfficientNet
import torch
model = EfficientNet.from_pretrained('efficientnet-b7')
img = torch.rand(1,3,224,224)
# ... image preprocessing as in the classification example ...
print(img.size()) # torch.Size([1, 3, 224, 224])

x,features = model.extract_features(img)
print(x.size()) # torch.Size([1, 1280, 7, 7])

# endpoints = model.extract_endpoints(img)
# print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
# print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
# print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
# print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
# print(endpoints['reduction_5'].shape)  # torch.Size([1, 320, 7, 7])
# print(endpoints['reduction_6'].shape)  # torch.Size([1, 1280, 7, 7])