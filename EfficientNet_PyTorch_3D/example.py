from efficientnet_pytorch_3d import EfficientNet3D
import torch
from torchsummary import summary

device = torch.device("cuda")

# model = EfficientNet3D.from_name("efficientnet-b7").to(device)
model = EfficientNet3D.from_name("efficientnet-b7", in_channels=1).to(device)
# summary(model, input_size=(1, 224, 224, 224))

# model = model.to(device)
inputs = torch.randn((1, 1, 96, 96, 96)).to(device)
outputs, features = model.extract_features(inputs)
print("outputs:",outputs.size())
for feature in features:
    print(feature.size())
# labels = torch.tensor([0]).to(device)
# # test forward
# num_classes = 2

# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# model.train()
# for epoch in range(2):
#     # zero the parameter gradients
#     optimizer.zero_grad()

#     # forward + backward + optimize
#     outputs = model(inputs)
#     loss = criterion(outputs, labels)
#     loss.backward()
#     optimizer.step()

#     # print statistics
#     print('[%d] loss: %.3f' % (epoch + 1, loss.item()))

# print('Finished Training')
