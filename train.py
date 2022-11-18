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
import torch.nn.functional as F

from torch.optim import RMSprop



# %%
class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, data_class: str, batch_size = 1024, data_dir: str = "./data"):
        super().__init__()
        self.data_dir = data_dir
        self.data_class = CIFAR10 if data_class == '10' else CIFAR100 if data_class == '100' else None
        self.batch_size = batch_size
        
        if self.data_class is None:
            raise ValueError('data_class must be 10 or 100')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
            transforms.RandAugment(magnitude = 5),
        ])
        

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



#%%
class TrainModule(pl.LightningModule):
    def __init__(self, model, lr=1e-3, warmup = False, **kwargs):
        super().__init__()
        self.model = model
        self.lr = lr
        self.warmup = warmup
        

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)

        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y).item() / len(y)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)

        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y).item() / len(y)
        self.log('val_acc', acc)
        

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y).item() / len(y)
        self.log('test_acc', acc)


    def configure_optimizers(self):
        return RMSprop(self.parameters(), lr=self.lr, alpha=0.99, eps=1e-08, weight_decay=0.9, momentum=0.9, centered=False)

    # warmup
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # warm up lr
        if self.warmup:
            if self.trainer.global_step < self.warmup:
                lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmup)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_scale * self.lr
        # update params
        optimizer.step(closure=optimizer_closure)
        # update lr
        optimizer.zero_grad()
#         self.lr_scheduler.step()



#%%

from models import make_efficientnetv2

if __name__ == '__main__':

    
    
    args = {
        'data' : '100',
        'warmup' : 10,
        'epoch' : 50,
        'dropout_rate' : 0.1,
    }
    
    name = f'ENv2-s CIFAR{args["data"]}'
    
    for k, v in args.items():
        if k == 'data' : 
            continue
            
        if v :
            if v is True :
                name = name + ' ' + k
            else :
                name = name + ' ' + f'{k}={v}'
    
    print(name)
    
    
    cifar = CIFARDataModule(data_class=args['data'])
    model = make_efficientnetv2('s', num_classes=int(args['data']), dropout_rate=args['dropout_rate'])


    logger = TensorBoardLogger('tb_logs', name=name)


    trainer = Trainer(
        gpus=1,
        max_epochs=args['epoch'],
        logger=logger,
    )

    train_module = TrainModule(model, warmup=args['warmup'])
    trainer.fit(train_module, cifar)
    
    trainer.test(train_module, cifar)
