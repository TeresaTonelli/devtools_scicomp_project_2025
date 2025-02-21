import torch
import torch.nn as nn
import lightning.pytorch as pl
import torchmetrics
from .model import AlexNet
class Classifier(pl.LightningModule):

	def __init__(self, model):
		super().__init__()
		self.model = model
		self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.model.num_classes)
		self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.model.num_classes)
		self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.model.num_classes)

	def _classifier_step(self, batch):
		x, y = batch 
		y_hat = self.model(x)
		loss = nn.functional.cross_entropy(y_hat, y)
		return y_hat, y, loss

	def training_step(self, batch):
		y_hat, y, loss = self._classifier_step(batch)
		self.train_accuracy(y_hat, y)
		self.log('train_accuracy', self.train_accuracy, on_step=True, on_epoch=False)
		return loss

	def validation_step(self, batch):
		y_hat, y, loss = self._classifier_step(batch)
		self.val_accuracy(y_hat, y)
		self.log('val_accuracy', self.val_accuracy, on_step=True, on_epoch=False)

	def test_step(self, batch):
		y_hat, y, loss = self._classifier_step(batch)
                self.test_accuracy(y_hat, y)
                self.log('val_accuracy', self.test_accuracy, on_step=True, on_epoch=False)


	def forward(self, x):
		self.model(x)


	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
		return optimizer
