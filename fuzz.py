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
        model, testloader, device, susp, num_test, popsize,
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

            # logging the adversarial samples
            err_idx = (~comp).nonzero().flatten().tolist()
            for eid in err_idx:
                p = predicted[eid].item()
                t = targets[eid].item()
                logging.info('%s | %s', descs[eid], f'predict: {p}, target: {t}')

            # statistic converage
            if susp is None: continue
            for i, t in enumerate(targets):
                susp_l2chn = susp[str(t.item())]
                for lname, chns in susp_l2chn.items():
                    if len(chns) == 0:
                        nconv.append(0)
                    else:
                        act_info, n = conv_info[lname]
                        sum_act = act_info[i][chns].sum().item()
                        conv = sum_act / (n * len(chns))
                        nconv.append(conv)

    if susp is None:
        return None

    mconv, _ = torch.tensor(nconv).view(num_test, popsize, -1).max(dim=-1)
    return mconv


def _forward_conv(lname):
    def __hook(module, finput, foutput):
        b, c, *_ = foutput.size()
        squeeze = foutput.view(b, c, -1)
        n = squeeze.size(-1)
        actives = (squeeze > 0).sum(dim=-1).cpu()
        global conv_info
        conv_info[lname] = (actives, n)
    return __hook


def main():
    opt = parser.parse_args()
    print(opt)
    guard_folder(opt)

    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        filename=os.path.join(
            opt.output_dir, opt.dataset, opt.model, f'adversarial_samples_{opt.politice}.log'
        ),
        filemode='w',
        level=logging.INFO
    )

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    model = load_model(opt).to(device)

    if opt.politice == 'conv':
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                m.register_forward_hook(_forward_conv(n))
        susp = load_select_info(opt)
    else:
        susp = None

    testset = load_dataset(opt)
    num_test = len(testset)
    gene = GenericSearcher(opt, num_test=num_test)

    for _ in range(opt.fuzz_epoch):
        mutators = gene.generate_next_population()
        testloader = generate_test_dataset(opt, testset, mutators)
        mconv = test(model, testloader, device, susp, num_test, opt.popsize)
        gene.fitness(mconv)

    print('[info] Done.')


if __name__ == "__main__":
    main()

