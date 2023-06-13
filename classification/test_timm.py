import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import timm
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
# model = timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True)
# num_ftrs = model.head.in_features
# model.head = torch.nn.Linear(num_ftrs, 10)  # CIFAR10 has 10 classes

extractor = AutoFeatureExtractor.from_pretrained("jadohu/BEiT-finetuned")
model = AutoModelForImageClassification.from_pretrained("jadohu/BEiT-finetuned")
model = model.to(device)

# data_norm_mean, data_norm_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
data_norm_mean, data_norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=data_norm_mean, std=data_norm_std)
     ])

# Load CIFAR10 data
trainset = torchvision.datasets.CIFAR10(root='./data/datasets/images/cifar10', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data/datasets/images/cifar10', train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=24, shuffle=True)
testloader = DataLoader(testset, batch_size=24, shuffle=False)

# Specify the loss function and the optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
# model.train()
# for epoch in range(10):  # loop over the dataset multiple times
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data[0].to(device), data[1].to(device)
#         # zero the parameter gradients
#         optimizer.zero_grad()
#         # forward + backward + optimize
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#     print(f"Epoch {epoch+1} completed")

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        # _, predicted = torch.max(outputs.data, 1)
        # total += labels.size(0)
        # correct += (predicted == labels).sum().item()
        _, predicted = torch.max(outputs.logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
