import collections
import torch
from typing import  Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm

from dlvc.wandb_logger import WandBLogger
from dlvc.dataset.oxfordpets import OxfordPetsCustom

class BaseTrainer(metaclass=ABCMeta):
    '''
    Base class of all Trainers.
    '''

    @abstractmethod
    def train(self) -> None:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float]:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float]:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

class ImgSemSegTrainer(BaseTrainer):
    """
    Class that stores the logic for training a model for image classification.
    """
    def __init__(self, 
                 model, 
                 optimizer,
                 loss_fn,
                 lr_scheduler,
                 train_metric,
                 val_metric,
                 train_data,
                 val_data,
                 device,
                 num_epochs: int, 
                 training_save_dir: Path,
                 batch_size: int = 4,
                 val_frequency: int = 5):
        '''
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.SegMetrics): SegMetrics class to get mIoU of training set
            val_metric (dlvc.metrics.SegMetrics): SegMetrics class to get mIoU of validation set
            train_data (dlvc.datasets...): Train dataset
            val_data (dlvc.datasets...): Validation dataset
            device (torch.device): cuda or cpu - device used to train the network
            num_epochs (int): number of epochs to train the network
            training_save_dir (Path): the path to the folder where the best model is stored
            batch_size (int): number of samples in one batch 
            val_frequency (int): how often validation is conducted during training (if it is 5 then every 5th 
                                epoch we evaluate model on validation set)

        What does it do:
            - Stores given variables as instance variables for use in other class methods e.g. self.model = model.
            - Creates data loaders for the train and validation datasets
            - Optionally use weights & biases for tracking metrics and loss: initializer W&B logger

        '''
        #recycled from task 1
    
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.num_epochs = num_epochs
        self.training_save_dir = training_save_dir
        self.batch_size = batch_size
        self.val_frequency = val_frequency
        self.subtract_one = isinstance(train_data, OxfordPetsCustom)

        self.num_train_data = len(self.train_data)
        self.num_val_data = len(self.val_data)


        #unique id for run
        model_name = model.net.__class__.__name__
        unique_id = str(id(model))[-6:]
        
        run_name = f"{model_name}_{unique_id}"
                                   
        self.wandb_logger = WandBLogger(model= model.net, run_name=run_name) if hasattr(WandBLogger, "log") else None

        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=4, pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False,num_workers=4, pin_memory=True)
        

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        """
        Training logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean IoU for this epoch.

        epoch_idx (int): Current epoch number
        """
        self.model.train()
        self.train_metric.reset() #reset metrics

        epoch_loss = 0.
        for batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch_idx}"): #some ui to see progress
            images, labels = batch
            labels = labels.squeeze(1)-int(self.subtract_one)
            batch_size = images.shape[0]


            self.optimizer.zero_grad() #reset gradients from other batches
            outputs = self.model(images.to(self.device))
            if isinstance(outputs, collections.OrderedDict):
                outputs = outputs['out']

            loss = self.loss_fn(outputs, labels.to(self.device))
            loss.backward()
            
            self.optimizer.step() #update weights

            epoch_loss += (loss.item() * batch_size)
            self.train_metric.update(outputs.detach().cpu(), labels.detach().cpu())

        self.lr_scheduler.step()
        epoch_loss /= self.num_train_data
        epoch_mIoU = self.train_metric.mIoU()
        
        print(f"______epoch {epoch_idx} \n")
        print(f"Loss: {epoch_loss}")
        print(self.train_metric)

        return epoch_loss, epoch_mIoU


    def _val_epoch(self, epoch_idx:int) -> Tuple[float, float]:
        """
        Validation logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean IoU for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """
        self.model.eval() #validation/test mode
        self.val_metric.reset() #reset metrics

        epoch_loss = 0.

        with torch.no_grad(): #no gradient now, only eval
            for batch in tqdm(self.val_loader, desc=f"Validation Epoch {epoch_idx}"):
                images, labels = batch
                labels = labels.squeeze(1)-int(self.subtract_one)
                batch_size = images.shape[0] 
    
                outputs = self.model(images.to(self.device))
                if isinstance(outputs, collections.OrderedDict):
                    outputs = outputs['out']

                    
                loss = self.loss_fn(outputs, labels.to(self.device))
                # Gather metrics
                epoch_loss += (loss.item() * batch_size)
                self.val_metric.update(outputs.cpu(), labels.cpu())
    

        epoch_loss /= self.num_val_data
        epoch_mIoU = self.val_metric.mIoU()
        print(f"______epoch {epoch_idx} - validation \n")
        print(f"Loss: {epoch_loss}")
        print(self.val_metric)

        return epoch_loss, epoch_mIoU

    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean IoU on validation data set is higher
        than currently saved best mean IoU or if it is end of training. 
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """
        best_miou = 0.0
        
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch} out of {self.num_epochs}")
            train_loss, train_miou = self._train_epoch(epoch)

            #logging for train

            if epoch % self.val_frequency != 0:
                if self.wandb_logger:
                    self.wandb_logger.log({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_mIoU": train_miou})


            else: #only val if frequency is reached
                val_loss, val_miou= self._val_epoch(epoch)

                #logging for validation
                if self.wandb_logger:
                    self.wandb_logger.log({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_mIoU": train_miou,
                        "val_loss": val_loss,
                        "val_mIoU": val_miou
                    })


                #checking result, saving best model if found
                if val_miou > best_miou:
                    print(f"New best model found, val_mIoU is: {val_miou:.4f})")
                    self.model.save(self.training_save_dir, suffix=type(self.model.net).__name__)
                    print(f"New best model saved\n")
                    best_miou = val_miou

    def dispose(self) -> None:
            self.wandb_logger.finish()
                





            
            


