import argparse
import shutil
import torch
import yaml
from torch import optim

import models
import utils
import os

def train(_config, resume: bool = False, test: bool = False):
    learning_rate = _config['learning_rate']
    weight_decay = _config['weight_decay']
    frozen_predictor = _config['frozen_predictor']
    device = torch.device(_config['device'])
    saved_folder = os.path.join(_config['saved_path'],_config['name'])
    if not resume and not test:
        model_path =os.path.join(_config['fixed_model'],'best_model.pkl')
        fixed_config_path = os.path.join(_config['fixed_model'],'config.yaml')
        with open(fixed_config_path) as f:
            fixed_config = yaml.safe_load(f)
        #improve current _config
        _config['data'] = fixed_config['data']
        _config['model']['predictor'] = fixed_config['model']['predictor']       
        shutil.rmtree(saved_folder, ignore_errors=True)
        os.makedirs(saved_folder) 
        with open(os.path.join(saved_folder, 'config.yaml'), 'w+') as _f:
            yaml.safe_dump(config, _f)       
    else:
        model_path = os.path.join(_config['saved_path'],_config['name'],'best_model.pkl')
        with open(os.path.join(_config['saved_path'],_config['name'], 'config.yaml')) as f:
            _config = yaml.safe_load(f)
    
    dataset = _config['data']['dataset']
   
    model = models.create_model(dataset, 
                                _config['model']['predictor'], 
                                _config['model']['graph_learner'], 
                                device=device)

    saved = torch.load(model_path) 
    model.load_state_dict(saved['model_state_dict'], strict=False)
    
    if frozen_predictor:
        model.graph_learner.adaptive.requires_grad_(False)        
        for param in model.predictor.parameters():
            param.requires_grad_(False)
    

    datasets = utils.get_datasets(dataset, _config['data']['input_dim'], _config['data']['output_dim'])

    scaler = utils.ZScoreScaler(datasets['train'].mean, datasets['train'].std)
    optimizer = optim.Adam([
        {'params': model.graph_learner.parameters()},
        {'params': model.predictor.parameters(), 'lr': 1e-5}  # fine-tune the predictor
    ], lr=learning_rate)
    loss = utils.get_loss('MaskedMAELoss')

    trainer = utils.OursTrainer(model, loss, scaler, device, optimizer, weight_decay, 2, 5)
    if not test:
        utils.train_model(
            datasets=datasets,
            batch_size=_config['data']['batch-size'],
            folder=saved_folder,
            trainer=trainer,
            scheduler=None,
            epochs=_config['epochs'],    
            early_stop_steps=_config['early_stop_steps']           
        )

    utils.test_model(
        datasets=datasets,
        batch_size=_config['data']['batch-size'],
        trainer=trainer,
        folder=saved_folder
    )


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