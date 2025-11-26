"""PyTorch Lightning module for classifier training."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score


class ClassifierModule(pl.LightningModule):
    """PyTorch Lightning module wrapper for classifier training."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer_config: dict,
        learning_rate: float = 1e-3,
    ):
        """
        Args:
            model: The classifier model (nn.Module)
            optimizer_config: Optimizer configuration dict with _target_ and _partial_
            learning_rate: Learning rate (will override optimizer_config.lr if present)
        """
        super().__init__()
        self.model = model
        self.optimizer_config = optimizer_config
        self.learning_rate = learning_rate
        
        # Metrics
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.train_precision = Precision(task="binary")
        self.val_precision = Precision(task="binary")
        self.train_recall = Recall(task="binary")
        self.val_recall = Recall(task="binary")
        self.train_f1 = F1Score(task="binary")
        self.val_f1 = F1Score(task="binary")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        
        # Update metrics
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.train_precision(preds, y)
        self.train_recall(preds, y)
        self.train_f1(preds, y)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        
        # Update metrics
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.val_f1(preds, y)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer from config."""
        from hydra.utils import instantiate
        
        # Create optimizer with model parameters
        optimizer_config = self.optimizer_config.copy()
        optimizer_config["params"] = self.model.parameters()
        
        # Override lr if learning_rate is set
        if "lr" in optimizer_config:
            optimizer_config["lr"] = self.learning_rate
        
        optimizer = instantiate(optimizer_config)
        
        return optimizer


