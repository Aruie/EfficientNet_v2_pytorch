# %%
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision import transforms
import torch.functional as F


# %%

class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, data_class: str, data_dir: str = "./data"):
        super().__init__()
        self.data_dir = data_dir
        self.data_class = CIFAR10 if data_class == '10' else CIFAR100 if data_class == '100' else None
        if self.data_class is None:
            raise ValueError('data_class must be 10 or 100')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))])

    def prepare_data(self):
        self.data_class(self.data_dir, train=True, download=True)
        self.data_class(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            data_full = self.data_class(self.data_dir, train=True, transform=self.transform)
            self.data_train, self.data_val = random_split(data_full, [45000, 5000])

        if stage == "test":
            self.data_test = self.data_class(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.data_predict = self.data_class(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.data_predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        if stage == "fit":
            del self.data_train
            del self.data_val

        if stage == "test":
            del self.data_test

        if stage == "predict":
            del self.data_predict





# 학습
class TrainModule(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)



#%%

from models import make_efficientnetv2


if __name__ == '__main__':

    cifar = CIFARDataModule(data_class='10')
    model = make_efficientnetv2('s', num_classes=10)
    # %%

    train_module = TrainModule(model=model, lr=1e-3)
    # %%

    trainer = Trainer(gpus=0, max_epochs=10)

    # %%
    trainer.fit(train_module, cifar)
    # %%
    trainer.test(train_module, cifar)
