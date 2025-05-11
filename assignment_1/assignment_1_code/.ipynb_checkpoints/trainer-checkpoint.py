import torch
from typing import Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm

# for wandb users:
from assignment_1_code.wandb_logger import WandBLogger


class BaseTrainer(metaclass=ABCMeta):
    """
    Base class of all Trainers.
    """

    @abstractmethod
    def train(self) -> None:
        """
        Holds training logic.
        """

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float, float]:
        """
        Holds validation logic for one epoch.
        """

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float, float]:
        """
        Holds training logic for one epoch.
        """

        pass


class ImgClassificationTrainer(BaseTrainer):
    """
    Class that stores the logic for training a model for image classification.
    """

    def __init__(
        self,
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
        val_frequency: int = 5,
    ) -> None:
        """
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of training set
            val_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of validation set
            train_data (dlvc.datasets.cifar10.CIFAR10Dataset): Train dataset
            val_data (dlvc.datasets.cifar10.CIFAR10Dataset): Validation dataset
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

        """

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
                                   
        self.wandb_logger = WandBLogger() if hasattr(WandBLogger, "log") else None

        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Training logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch.

        epoch_idx (int): Current epoch number
        """
        self.model.train()
        self.train_metric.reset()

        total_loss = 0.0
        for batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch_idx}"):
            images, labels = batch[0].to(self.device), batch[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            self.train_metric.update(outputs, labels)
            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)

        print(f"Loss: {avg_loss:.4f}")
        print(str(train_metric))

        return avg_loss, train_metric.accuracy(), train_metric.per_class_accuracy()


    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Validation logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """
        self.model.eval()
        self.val_metric.reset()

        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Validation Epoch {epoch_idx}"):
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
    
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
    
                self.val_metric.update(outputs, labels)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)

        print(f"EPOCH {epoch_idx}")
        print(f"Loss: {avg_loss:.4f}")
        print(str(val_metric))

        return avg_loss, val_metric.accuracy(), val_metric.per_class_accuracy()

    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean per class accuracy on validation data set is higher
        than currently saved best mean per class accuracy.
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """
        for epoch in range(1, self.num_epochs + 1):
            print(f"\n=== Epoch {epoch}/{self.num_epochs} ===")
            train_loss, train_acc, train_pc_acc = self._train_epoch(epoch)
    
            if epoch % self.val_frequency == 0 or epoch == self.num_epochs:
                val_loss, val_acc, val_pc_acc = self._val_epoch(epoch)
    
                # Log to wandb if available
                if self.wandb_logger:
                    self.wandb_logger.log({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "train_pc_acc": train_pc_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_pc_acc": val_pc_acc,
                    })
    
                # Save model if per-class accuracy improves
                if val_pc_acc > best_pc_acc:
                    print(f"New best model found (mPCAcc: {val_pc_acc:.4f}), saving...")
                    self.model.save(self.training_save_dir, suffix="best")
                    best_pc_acc = val_pc_acc