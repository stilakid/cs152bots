import torch
import torch.nn.functional as F
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchmetrics import Accuracy
import pytorch_lightning as pl


class NN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=True)
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # modify final linear layer
        self.model.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)

        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("val_loss", loss)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("test_loss", loss)

        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, y)
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-7,7)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))]
    )

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    trainset, valset = random_split(trainset, [45000, 5000])

    if torch.cuda.is_available():
        batch_size = 32
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2, prefetch_factor=4, shuffle=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, num_workers=2, prefetch_factor=4, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=2, prefetch_factor=4, shuffle=False)
    else:
        batch_size = 32
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=0, prefetch_factor=None, shuffle=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, num_workers=0, prefetch_factor=None, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=0, prefetch_factor=None, shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = NN()
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k = 1,
        verbose = True, 
        monitor = "val_loss",
        mode = "min",
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00, 
        patience=10, 
        verbose=False, 
        mode="min",
    )

    trainer = pl.Trainer(max_epochs=100, callbacks = [checkpoint_callback, early_stop_callback])
    trainer.fit(model, trainloader, valloader)
    trainer.test(model, testloader)
