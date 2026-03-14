import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy

EPSILON = 1e-8


class EWCDR(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.pre_dict = {}
        self.target_dict = {}
        self.omega = None
        self.mean = None

        self.init_epochs = self.args.get("init_epochs", 200)
        self.init_lr = self.args.get("init_lr", 0.1)
        self.init_milestones = self.args.get("init_milestones", [60, 120, 170])
        self.init_lr_decay = self.args.get("init_lr_decay", 0.1)
        self.init_weight_decay = self.args.get("init_weight_decay", 5e-4)
        self.epochs = self.args.get("epochs", 180)
        self.lr = self.args.get("lr", 0.1)
        self.milestones = self.args.get("milestones", [70, 120, 150])
        self.lr_decay = self.args.get("lr_decay", 0.1)
        self.weight_decay = self.args.get("weight_decay", 2e-4)
        self.batch_size = self.args.get("batch_size", 128)
        self.num_workers = self.args.get("num_workers", 4)
        self.omegamax = self.args.get("omegamax", 0.0001)
        self.lamda = self.args["lamda"]

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        self._network_module_ptr = self._network
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        self._train(self.train_loader, self.test_loader)

        if self.omega is None:
            self.omega = self.getImportance(self.train_loader)
        else:
            alpha = self._known_classes / self._total_classes
            new_omega = self.getImportance(self.train_loader)
            for n, p in new_omega.items():
                new_omega[n][: len(self.omega[n])] = (
                        alpha * self.omega[n]
                        + (1 - alpha) * new_omega[n][: len(self.omega[n])]
                )
            self.omega = new_omega

        self.mean = {
            n: p.clone().detach()
            for n, p in self._network.named_parameters()
            if p.requires_grad
        }

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.init_lr,
                weight_decay=self.init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.init_milestones, gamma=self.init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.milestones, gamma=self.lr_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        for epoch in range(self.init_epochs):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.init_epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.init_epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        for epoch in range(self.epochs):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss_clf = F.cross_entropy(
                    logits[:, self._known_classes:], targets - self._known_classes
                )

                loss_ewc = self.lamda * self.compute_ewc()
                loss = loss_clf + loss_ewc

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            logging.info(info)

    def compute_ewc(self):
        loss = 0
        if len(self._multiple_gpus) > 1:
            for n, p in self._network.module.named_parameters():
                if n in self.omega.keys():
                    loss += (
                            torch.sum(
                                (self.omega[n])
                                * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                            )
                            / 2
                    )
        else:
            for n, p in self._network.named_parameters():
                if n in self.omega.keys():
                    loss += (
                            torch.sum(
                                (self.omega[n])
                                * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                            )
                            / 2
                    )
        return loss

    def getImportance(self, train_loader):
        omega = {
            n: torch.zeros(p.shape).to(self._device)
            for n, p in self._network.named_parameters()
            if p.requires_grad
        }
        self._network.train()
        optimizer = optim.SGD(self._network.parameters(), lr=self.lr)
        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            logits = self._network(inputs)["logits"]
            logits = logits*-1 #LR
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            for n, p in self._network.named_parameters():
                if p.grad is not None:
                    omega[n] += p.grad.pow(2).clone()
        for n, p in omega.items():
            omega[n] = p / len(train_loader)
            omega[n] = torch.min(omega[n], torch.tensor(self.omegamax).to(self._device))
        return omega