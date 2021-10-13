import copy
import json
import os
import pickle
import time
from typing import Dict, Optional, List

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .data import get_dataloaders, TrafficPredictionDataset
from .evaluate import evaluate


class OursTrainer(object):
    def __init__(self,
                 model: nn.Module,
                 loss: nn.Module,
                 scaler,
                 device: torch.device,
                 optimizer,
                 reg_weight_decay: float,
                 reg_norm: int,
                 max_grad_norm: Optional[float]):
        self.model = model.to(device)
        self.loss = loss.to(device)
        self.optimizer = optimizer
        self.scaler = scaler
        self.clip = max_grad_norm
        self.device = device
        self.norm = reg_norm
        self.weight_decay = reg_weight_decay

    def train(self, inputs, targets):
        self.model.train()
        self.optimizer.zero_grad()

        predicts = self._run(inputs)
        loss = self.loss(predicts, targets.to(self.device))
        reg = get_regularization(self.model, weight_decay=self.weight_decay, p=self.norm) #加入正则项
        (loss + reg).backward()
        if self.clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        return predicts

    def predict(self, inputs):
        self.model.eval()
        return self._run(inputs)

    def _run(self, inputs):
        inputs[..., 0] = self.scaler.transform(inputs[..., 0], 0.0)

        outputs = self.model(inputs.to(self.device))
        return self.scaler.inverse_transform(outputs, 0.0)

    def load_state_dict(self, model_state_dict, optimizer_state_dict):
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.model = self.model.to(self.device)
        set_device_recursive(self.optimizer.state, self.device)
        '''
        print('Resume:load_state_dict__________________________')
        #print model's state_dict
        print('Model.state_dict:')
        for param_tensor in self.model.state_dict():
            #打印 key value字典
            print(param_tensor,'\t',self.model.state_dict()[param_tensor].size())
 
        #print optimizer's state_dict
        print('Optimizer,s state_dict:')
        for var_name in self.optimizer.state_dict():
            print(var_name,'\t',self.optimizer.state_dict()[var_name])
        '''



def train_model(
        datasets: Dict[str, TrafficPredictionDataset],
        batch_size: int,
        folder: str,
        trainer: OursTrainer,
        scheduler,
        epochs: int,
        early_stop_steps: int ):#Optional[int]
    data_loaders = get_dataloaders(datasets, batch_size)

    save_path = os.path.join(folder, 'best_model.pkl')
    begin_epoch = 0

    if os.path.exists(save_path):
        save_dict = torch.load(save_path)

        trainer.load_state_dict(save_dict['model_state_dict'], save_dict['optimizer_state_dict'])

        best_val_loss = save_dict['best_val_loss']
        begin_epoch = save_dict['epoch'] + 1
    else:
        save_dict = dict()
        best_val_loss = float('inf')
        begin_epoch = 0

    phases = ['train', 'val', 'test']

    writer = SummaryWriter(folder)

    since = time.perf_counter()

    print(trainer.model)
    print(f'Trainable parameters: {get_number_of_parameters(trainer.model)}.')

    try:
        for epoch in range(begin_epoch, begin_epoch + epochs):
            running_metrics = dict()
            for phase in phases:
                steps, predicts, targets = 0, list(), list()
                for x, y in tqdm(data_loaders[phase], f'{phase.capitalize():5} {epoch}'):
                    targets.append(y.numpy().copy())
                    if phase == 'train':
                        y_ = trainer.train(x, y)
                    else:
                        with torch.no_grad():
                            y_ = trainer.predict(x)

                    predicts.append(y_.detach().cpu().numpy())

                    # For the issue that the CPU memory increases while training. DO NOT know why, but it works.
                    torch.cuda.empty_cache()

                # 性能
                running_metrics[phase] = evaluate(np.concatenate(predicts), np.concatenate(targets))

                if phase == 'val':
                    if running_metrics['val']['loss'] < best_val_loss:
                        best_val_loss = running_metrics['val']['loss']
                        save_dict.update(
                            model_state_dict=copy.deepcopy(trainer.model.state_dict()),
                            epoch=epoch,
                            best_val_loss=best_val_loss,
                            optimizer_state_dict=copy.deepcopy(trainer.optimizer.state_dict()))
                        save_model(save_path, **save_dict)
                        print(f'A better model at epoch {epoch} recorded.')
                    elif early_stop_steps is not None and epoch - save_dict['epoch'] > early_stop_steps:
                        raise ValueError('Early stopped.')

            loss_dict = {f'{phase} loss': running_metrics[phase].pop('loss') for phase in phases}
            print(loss_dict)

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(loss_dict['train'])
                else:
                    scheduler.step()

            writer.add_scalars('Loss', loss_dict, global_step=epoch)
            for metric in running_metrics['train'].keys():
                for phase in phases:
                    for key, val in running_metrics[phase][metric].items():
                        writer.add_scalars(f'{metric}/{key}', {f'{phase}': val}, global_step=epoch)
    except (ValueError, KeyboardInterrupt) as e:
        print(e)
    time_elapsed = time.perf_counter() - since
    print(f"cost {time_elapsed} seconds")
    print(f'The best adaptor and model of epoch {save_dict["epoch"]} successfully saved at `{save_path}`')


def test_model(
        datasets: Dict[str, TrafficPredictionDataset],
        batch_size: int,
        trainer: OursTrainer,
        folder: str):
    dataloaders = get_dataloaders(datasets, batch_size)

    saved_path = os.path.join(folder, 'best_model.pkl')

    saved_dict = torch.load(saved_path)
    trainer.model.load_state_dict(saved_dict['model_state_dict'])

    predictions, running_targets = list(), list()
    with torch.no_grad():
        for inputs, targets in tqdm(dataloaders['test'], 'Test model'):
            running_targets.append(targets.numpy().copy())
            predicts = trainer.predict(inputs)
            predictions.append(predicts.cpu().numpy())

    # 性能
    predictions, running_targets = np.concatenate(predictions), np.concatenate(running_targets)
    scores = evaluate(predictions, running_targets)
    scores.pop('loss')
    print('test results:')
    print(json.dumps(scores, cls=JsonEncoder, indent=4))
    with open(os.path.join(folder, 'test-scores.json'), 'w+') as f:
        json.dump(scores, f, cls=JsonEncoder, indent=4)

    np.savez(os.path.join(folder, 'test-results.npz'), predictions=predictions, targets=running_targets)


def save_model(path: str, **save_dict):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    torch.save(save_dict, path)


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)


def get_number_of_parameters(model: nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def get_scheduler(name, optimizer, **kwargs):
    return getattr(optim.lr_scheduler, name)(optimizer, **kwargs)


def get_optimizer(name: str, parameters, **kwargs):
    return getattr(optim, name)(parameters, **kwargs)


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def set_device_recursive(var, device):
    for key in var:
        if isinstance(var[key], dict):
            var[key] = set_device_recursive(var[key], device)
        else:
            try:
                var[key] = var[key].to(device)
            except AttributeError:
                pass
    return var


def set_requires_grad(models: List[nn.Module], required: bool):
    for model in models:
        for param in model.parameters():
            param.requires_grad_(required)


def get_regularization(model: nn.Module, weight_decay: float, p: float = 2.0):
    weight_list = list(filter(lambda item: 'weight' in item[0], model.named_parameters()))
    return weight_decay * sum(torch.norm(w, p=p) for name, w in weight_list)
