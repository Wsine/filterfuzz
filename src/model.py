import os

import torch


def load_model(opt):
    if 'cifar' in opt.dataset:
        model_hub = 'chenyaofo/pytorch-cifar-models'
        model_name = f'{opt.dataset}_{opt.model}'
        model = torch.hub.load(model_hub, model_name, pretrained=True)
        return model
    elif opt.model == 'convstn':
        from models.gtsrb.convstn import Net
        model = Net()
        ckp = torch.load(
            os.path.join('models', 'gtsrb', 'model_40.pth'),
            map_location=torch.device('cpu')
        )
        model.load_state_dict(ckp)
        return model
    else:
        raise ValueError('Invalid dataset name')

