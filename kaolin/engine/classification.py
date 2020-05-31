# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional, Union

import torch

from .engine import Engine


class ClassificationEngine(Engine):
    r"""An ``Engine`` that runs the training/validation/inference process for
    a classification task.

    Example:
        >>> engine = ClassificationEngine(model, train_loader, val_loader)

    """

    def __init__(self, model: torch.nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 hyperparams: Optional[Dict] = {},
                 engineparams: Optional[Dict] = {},
                 criterion: Optional[torch.nn.Module] = None,
                 device: Optional[Union[torch.device, str]] = None
                 ):
        # Model to be trained.
        self.model = model
        # Train dataloader.
        self.train_loader = train_loader
        # Validataion dataloader.
        self.val_loader = val_loader
        # Hyperaparameters (learning rate, momentum, etc.).
        self.hyperparams = self.initialize_hyperparams(hyperparams)
        # Engine parameters (number of epochs to train for, etc.)
        self.engineparams = self.initialize_engineparams(engineparams)
        # Optimizer.
        if optimizer is None:
            self.optimizer = self.initialize_optimizer()
        else:
            self.optimizer = optimizer
        # Loss function to use
        if criterion is None:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
        # Device
        if device is None:
            self.device = 'cpu'
        else:
            self.device = device

        # Cast model to device.
        self.model = self.model.to(self.device)

        """
        Engine stats.
        """

        self.current_epoch = 0
        # Train loss/accuracy for the current epoch.
        self.train_loss_cur_epoch = 0.
        self.train_accuracy_cur_epoch = 0.
        # Validation loss/accuracy for the current epoch.
        self.val_loss_cur_epoch = 0.
        self.val_accuracy_cur_epoch = 0.
        # Number of minibatches thus far, in the current epoch.
        # This variable is reused across train/val phases.
        self.batch_idx = 0
        # Train loss/accuracy history.
        self.train_loss = []
        self.train_accuracy = []
        # Validation loss/accuracy history.
        self.val_loss = []
        self.val_accuracy = []

    def initialize_hyperparams(self, hyperparams: Dict):
        r"""Initializes hyperparameters for the training process. Uses defaults
        wherever user specifications are unavailable.

        Args:
            hyperparams (dict): User-specified hyperparameters ('lr', 'beta1',
                'beta2' for Adam).

        """
        paramdict = {}

        if 'lr' in hyperparams:
            paramdict['lr'] = hyperparams['lr']
        else:
            paramdict['lr'] = 1e-3
        if 'beta1' in hyperparams:
            paramdict['beta1'] = hyperparams['beta1']
        else:
            paramdict['beta1'] = 0.9
        if 'beta2' in hyperparams:
            paramdict['beta2'] = hyperparams['beta2']
        else:
            paramdict['beta2'] = 0.999

        return paramdict

    def initialize_engineparams(self, engineparams: Dict):
        r"""Initializes engine parameters. Uses defaults wherever user
        specified values are unavilable.

        Args:
            engineparams (dict): User-specified engine parameters ('epochs',
                'validate-every', 'print-every', 'save', 'savedir').
                'epochs': number of epochs to train for.
                'validate-every': number of epochs after which validation
                    occurs.
                'print-every': number of iterations (batches) after which to
                    keep printing progress to stdout.
                'save': whether or not to save trained models.
                'savedir': directory to save trained models to.

        """
        paramdict = {}

        if 'epochs' in engineparams:
            paramdict['epochs'] = engineparams['epochs']
        else:
            paramdict['epochs'] = 10
        if 'validate-every' in engineparams:
            paramdict['validate-every'] =\
                engineparams['validate-every']
        else:
            paramdict['validate-every'] = 2
        if 'print-every' in engineparams:
            paramdict['print-every'] = engineparams['print-every']
        else:
            paramdict['print-every'] = 10
        if 'save' in engineparams:
            paramdict['save'] = engineparams['save']
        else:
            paramdict['save'] = False
        # Currently, we do not set a default 'savedir'.
        if 'savedir' in engineparams:
            paramdict['savedir'] = engineparams['savedir']

        return paramdict

    def initialize_optimizer(self):
        r"""Initializes the optimizer. By default, uses the Adam optimizer with
        the specified hyperparameter for learning-rates, beta1, and beta2.

        """
        return torch.optim.Adam(self.model.parameters(),
                                lr=self.hyperparams['lr'],
                                betas=(self.hyperparams['beta1'],
                                       self.hyperparams['beta2']))

    def train_batch(self, batch):
        r"""Train for one mini-batch.

        Args:
            batch (tuple): One mini-batch of training data.

        """
        data, attr = batch
        data = data.to(self.device)
        label = attr['category'].to(self.device)
        pred = self.model(data)
        loss = self.criterion(pred, label.view(-1))
        self.train_loss_cur_epoch += loss.item()
        predlabel = torch.argmax(pred, dim=1)
        accuracy = torch.mean((predlabel == label.view(-1)).float())
        self.train_accuracy_cur_epoch += accuracy.detach().cpu().item()
        self.batch_idx += 1
        return loss, accuracy

    def validate_batch(self, batch):
        r"""Validate for one mini-batch.

        Args:
            batch (tuple): One mini-batch of validateion data.

        """
        data, attr = batch
        data = data.to(self.device)
        label = attr['category'].to(self.device)
        pred = self.model(data)
        loss = self.criterion(pred, label.view(-1))
        self.val_loss_cur_epoch += loss.item()
        predlabel = torch.argmax(pred, dim=1)
        accuracy = torch.mean((predlabel == label.view(-1)).float())
        self.val_accuracy_cur_epoch += accuracy.detach().cpu().item()
        self.batch_idx += 1
        return loss, accuracy

    def optimize_batch(self, loss):
        r"""Optimize model parameters for one mini-batch.

        Args:
            loss: training loss computed over the mini-batch.
        """
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def fit(self):
        r"""Train the model using the specified engine parameters.
        """

        for epoch in range(self.engineparams['epochs']):

            # Train phase
            
            self.train_loss_cur_epoch = 0.
            self.train_accuracy_cur_epoch = 0.
            self.batch_idx = 0

            self.model.train()

            for idx, batch in enumerate(self.train_loader):
                loss, accuracy = self.train_batch(batch)
                self.optimize_batch(loss)
                self.print_train_stats()
            self.train_accuracy.append(self.train_accuracy_cur_epoch\
                / self.batch_idx)

            # Validation phase

            self.val_loss_cur_epoch = 0.
            self.val_accuracy_cur_epoch = 0.
            self.batch_idx = 0

            self.model.eval()

            with torch.no_grad():
                for idx, batch in enumerate(self.val_loader):
                    loss, accuracy = self.validate_batch(batch)
                    self.print_validation_stats()
                self.val_accuracy.append(self.val_accuracy_cur_epoch\
                    / self.batch_idx)

            self.current_epoch += 1

    def print_train_stats(self):
        r"""Print current stats.
        """
        print('Epoch: {0}, Train loss: {1}, Train accuracy: {2}'.format(
            self.current_epoch, self.train_loss_cur_epoch / self.batch_idx,
            self.train_accuracy_cur_epoch / self.batch_idx))

    def print_validation_stats(self):
        r"""Print current stats.
        """
        print('Epoch: {0}, Val loss: {1}, Val accuracy: {2}'.format(
            self.current_epoch, self.val_loss_cur_epoch / self.batch_idx,
            self.val_accuracy_cur_epoch / self.batch_idx))
