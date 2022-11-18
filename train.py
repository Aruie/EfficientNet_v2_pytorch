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

from torch.optim import RMSprop, Adam

from argparse import ArgumentParser

from models import make_efficientnetv2



# %%
class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, data: str, batch_size, randaug_magnitude, data_dir: str = "./data"):
        super().__init__()
        self.data_dir = data_dir
        self.data_class = CIFAR10 if data == 'CIFAR10' else CIFAR100 if data == 'CIFAR100' else None
        self.batch_size = batch_size
        
        if self.data_class is None:
            raise ValueError(f'Invalid Data {data}')
            
        self.transform = transforms.Compose([
            transforms.RandAugment(magnitude = randaug_magnitude),
            transforms.ToTensor(),
            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
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
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        print('----------------------------------------')
        print('--- Hyper Parameters -------------------')
        print('----------------------------------------')
        print(self.hparams)
        print('----------------------------------------')

        num_classes = 10 if self.hparams['data'] == 'CIFAR10' else 100 if self.hparams['data'] == 'CIFAR100' else None
        if num_classes is None:
            raise ValueError('data_class must be 10 or 100')
        self.model = make_efficientnetv2('s', num_classes=num_classes, dropout_rate=self.hparams['dropout_rate'])

        self.lr = self.hparams['lr']
        self.warmup = self.hparams['warmup']
        

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
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.hparams['weight_decay'])

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


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('TrainModule')
        parser.add_argument('--data', type=str, default='CIFAR10', help='CIFAR10 or CIFAR100')
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        parser.add_argument('--warmup', type=int, default=0, help='warmup steps')
        parser.add_argument('--dropout', dest = 'dropout_rate', type=float, default=0.1)
        parser.add_argument('--decay', dest = 'weight_decay', type=float, default=0.0)
        return parent_parser



#%%



if __name__ == '__main__':

    
    parser = ArgumentParser()
    parser.add_argument('-e', dest='epoch', type=int, default=10, help='number of epochs')
    parser.add_argument('-b', dest='batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--rand_mag', dest = 'randaug_magnitude', type=int, default=5, help='RandAug magnitude')

    parser = TrainModule.add_model_specific_args(parser)
    args = parser.parse_args()

    
    dict_args = vars(args)

    name = f'ENv2-s {dict_args["data"]}'
    for k, v in dict_args.items():
        if k in  ['data', 'epoch'] : 
            continue
        if v :
            if v is True :
                name = name + ' ' + k
            else :
                name = name + ' ' + f'{k}={v}'
    
    print(name)
    
    cifar = CIFARDataModule(data=dict_args['data'], batch_size = dict_args['batch_size'], randaug_magnitude=dict_args['randaug_magnitude'])

    logger = TensorBoardLogger('tb_logs', name=name)
    trainer = Trainer(
        gpus = 1 if torch.cuda.is_available() else 0,
        # max_epochs=args['epoch'],
        max_epochs=2,
        logger=logger,
    )

    train_module = TrainModule(**dict_args)

    trainer.fit(train_module, cifar)
    trainer.test(train_module, cifar)
