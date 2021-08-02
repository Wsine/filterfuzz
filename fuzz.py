import logging

import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import generate_test_dataset, load_dataset
from generic_search import GenericSearcher
from model import load_model
from arguments import parser
from utils import *


conv_info = {}


def load_select_info(opt):
    fpath = os.path.join(opt.output_dir, opt.dataset, opt.model, 'susp_filters.json')
    with open(fpath, 'r') as f:
        susp = json.load(f)
    return susp


@torch.no_grad()
def test(
        opt, model, testloader, device, susp,
        desc="Evaluate", tqdm_leave=True):
    model.eval()

    global conv_info

    nconv = []
    correct, total = 0, 0
    with tqdm(testloader, desc=desc, leave=tqdm_leave) as tepoch:
        for inputs, descs, targets in tepoch:
            conv_info.clear()

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            total += targets.size(0)
            _, predicted = outputs.max(1)
            comp = predicted.eq(targets)
            correct += comp.sum().item()

            acc = 100. * correct / total
            tepoch.set_postfix(acc=acc)

            #  logging the adversarial samples
            err_idx = (~comp).nonzero().flatten().tolist()
            for eid in err_idx:
                p = predicted[eid].item()
                t = targets[eid].item()
                logging.info('%s | %s', descs[eid], f'predict: {p}, target: {t}')

            # statistic converage
            if opt.politice == 'random':
                continue
            for i, t in enumerate(targets):
                sum_act, sum_neu = 0, 0
                if opt.politice == 'negconv':
                    susp_l2chn = susp[str(t.item())]
                    for lname, chns in susp_l2chn.items():
                        if len(chns) == 0:
                            continue
                        act_info, num_neu = conv_info[lname]
                        sum_act += act_info[i][chns].sum().item()
                        sum_neu += (len(chns) * num_neu)
                else:  # neuconv
                    for lname in conv_info.keys():
                        act_info, num_neu = conv_info[lname]
                        sum_act += act_info[i].sum().item()
                        sum_neu += (len(act_info[i]) * num_neu)
                conv = sum_act / sum_neu if sum_neu > 0 else 0
                nconv.append(conv)

    if opt.politice == 'random':
        return None

    #  mconv, _ = torch.tensor(nconv).view(opt.num_test, opt.popsize, -1).max(dim=-1)
    mconv = torch.tensor(nconv).view(opt.num_test, opt.popsize, -1)
    return mconv


def _forward_conv(lname):
    def __hook(module, finput, foutput):
        global conv_info
        b, c, *_ = foutput.size()
        squeeze = foutput.view(b, c, -1)
        num_neu = squeeze.size(-1)
        actives = (squeeze > 0).sum(dim=-1).cpu()
        conv_info[lname] = (actives, num_neu)
    return __hook


def main():
    opt = parser.parse_args()
    print(opt)
    guard_options(opt)

    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        filename=os.path.join(
            opt.output_dir, opt.dataset, opt.model,
            f'adversarial_samples_{opt.politice}_g{opt.gpu_id}.log'
        ),
        filemode='w',
        level=logging.INFO
    )

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    model = load_model(opt).to(device)

    if opt.politice.endswith('conv'):
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                m.register_forward_hook(_forward_conv(n))
    if opt.politice == 'negconv':
        susp = load_select_info(opt)
    else:
        susp = None

    testset = load_dataset(opt)
    opt.num_test = len(testset)
    gene = GenericSearcher(opt, num_test=opt.num_test)

    for e in range(opt.fuzz_epoch):
        print('fuzz epoch =', e)
        mutators = gene.generate_next_population()
        testloader = generate_test_dataset(opt, testset, mutators)
        mconv = test(opt, model, testloader, device, susp)
        gene.fitness(mconv)

    print('[info] Done.')


if __name__ == "__main__":
    main()

