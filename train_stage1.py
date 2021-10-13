import argparse
import json
import os
import shutil

import torch
import yaml

from models import create_model
import utils


def train(_config, resume: bool = False, test: bool = False):
    print(json.dumps(config, indent=4))

    device = torch.device(_config['device'])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device.index)
    device = torch.device(0)

    dataset = _config['data']['dataset']
    optimizer_name = _config['optimizer']['name']
    scheduler_name = _config['scheduler']['name']

    loss = utils.get_loss(_config['loss']['name']) 

    loss.to(device)

    model = create_model(dataset,
                         _config['model']['predictor'],
                         _config['model']['graph_learner'],                     
                         device)

    optimizer = utils.get_optimizer(optimizer_name, model.parameters(), **_config['optimizer'][optimizer_name])
    
    scheduler = None
    if scheduler_name is not None:
        scheduler = utils.get_scheduler(scheduler_name, optimizer, **_config['scheduler'][scheduler_name])
   
    save_folder = os.path.join('saves', dataset, _config['name'])

    if not resume and not test:
        shutil.rmtree(save_folder, ignore_errors=True)
        os.makedirs(save_folder)

    with open(os.path.join(save_folder, 'config.yaml'), 'w+') as _f:
        yaml.safe_dump(_config, _f)

    datasets = utils.get_datasets(dataset, _config['data']['input_dim'], _config['data']['output_dim'])

    scaler = utils.ZScoreScaler(datasets['train'].mean, datasets['train'].std)
    trainer = utils.OursTrainer(model, loss, scaler, device, optimizer, **_config['trainer'])

    if not test:
        utils.train_model(
            datasets=datasets,
            batch_size=_config['data']['batch-size'],
            folder=save_folder,
            trainer=trainer,
            scheduler=scheduler,
            epochs=config['epochs'],
            early_stop_steps=config['early_stop_steps']
        )

    utils.test_model(
        datasets=datasets,
        batch_size=_config['data']['batch-size'],
        trainer=trainer,
        folder=save_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='if to resume a trained model?')
    parser.add_argument('--test', action='store_true', default=False,
                        help='if in the test mode?')
    parser.add_argument('--name', required=True, type=str, help='The name of the folder where the model is stored.')

    args = parser.parse_args()

    with open(os.path.join('config', f'{args.config}.yaml')) as f:
        config = yaml.safe_load(f)
        config['name'] = args.name
    if args.resume:
        print(f'Resume to {config["name"]}.')
        train(config, resume=True)
    elif args.test:
        print(f'Test {config["name"]}.')
        train(config, test=True)
    else:
        train(config)
