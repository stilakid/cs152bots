import torch
import torchvision
import torchvision.transforms as transforms
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from classifier import NN


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))]
)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, num_workers=0, prefetch_factor=None, shuffle=False)

y_pred = []
y_true = []

model = NN.load_from_checkpoint("lightning_logs/version_0/checkpoints/epoch=30-step=43617.ckpt")
model.eval()

# iterate over test data
for inputs, labels in testloader:
    output = model(inputs) # Feed Network

    output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    output[np.logical_and(output != 3, output != 5)] = 2
    output[output == 3] = 1
    output[output == 5] -= 0
    y_pred.extend(output) # Save Prediction
        
    labels = labels.data.cpu().numpy()
    labels[np.logical_and(labels != 3, labels != 5)] = 2
    labels[labels == 3] = 1
    labels[labels == 5] -= 0
    y_true.extend(labels) # Save Truth

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes = ("Dog (Adult)", "Cat (CSAM)", "Other")

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (13, 10))
sn.heatmap(df_cm, annot=True)
plt.savefig('confusion_matrix.png')
