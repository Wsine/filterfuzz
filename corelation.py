import random

import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import generate_test_dataset, load_dataset
from generic_search import GenericSearcher
from model import load_model
from arguments import parser
from utils import *


conv_info = {}
adv_conv = []
std_conv = []


def load_select_info(opt):
    fpath = os.path.join(opt.output_dir, opt.dataset, opt.model, 'susp_filters.json')
    with open(fpath, 'r') as f:
        susp = json.load(f)
    return susp


@torch.no_grad()
def test(
        model, testloader, device, susp,
        desc="Evaluate", tqdm_leave=True):
    model.eval()

    global conv_info, adv_conv, std_conv

    correct, total = 0, 0
    with tqdm(testloader, desc=desc, leave=tqdm_leave) as tepoch:
        for inputs, _, targets in tepoch:
            conv_info.clear()

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            total += targets.size(0)
            _, predicted = outputs.max(1)
            comp = predicted.eq(targets)
            correct += comp.sum().item()

            acc = 100. * correct / total
            tepoch.set_postfix(acc=acc)

            err_idx = (~comp).nonzero().flatten().tolist()
            corr_idx = comp.nonzero().flatten().tolist()
            if len(err_idx) > len(corr_idx):
                err_idx = random.sample(err_idx, len(corr_idx))
            elif len(corr_idx) > len(err_idx):
                corr_idx = random.sample(corr_idx, len(err_idx))

            # statistic converage
            for e in err_idx:
                sum_act, sum_neu = 0, 0
                susp_l2chn = susp[str(targets[e].item())]
                if any([ len(chns) == 0 for chns in susp_l2chn.values() ]):
                    continue
                for lname, chns in susp_l2chn.items():
                    act_info, num_neu = conv_info[lname]
                    sum_act += act_info[e][chns].sum().item()
                    sum_neu += (len(chns) * num_neu)
                conv = sum_act / sum_neu if sum_neu > 0 else 0
                adv_conv.append(conv)
            for c in corr_idx:
                sum_act, sum_neu = 0, 0
                susp_l2chn = susp[str(targets[c].item())]
                if any([ len(chns) == 0 for chns in susp_l2chn.values() ]):
                    continue
                for lname, chns in susp_l2chn.items():
                    act_info, num_neu = conv_info[lname]
                    sum_act += act_info[c][chns].sum().item()
                    sum_neu += (len(chns) * num_neu)
                conv = sum_act / sum_neu if sum_neu > 0 else 0
                std_conv.append(conv)

    return None


def _forward_conv(lname):
    def __hook(module, finput, foutput):
        global conv_info
        b, c, *_ = foutput.size()
        squeeze = foutput.view(b, c, -1)
        num_neu = squeeze.size(-1)
        actives = (squeeze > 0).sum(dim=-1).cpu()
        conv_info[lname] = (actives, num_neu)
    return __hook


def show_statisitc():
    global adv_conv, std_conv
    print('adversarial statistic')
    print('num of samples: {}'.format(len(adv_conv)))
    print('max of negative converage: {:.2f}'.format(max(adv_conv)))
    print('mean of negative converage: {:.2f}'.format(sum(adv_conv) / len(adv_conv)))
    print('min of negative converage: {:.2f}'.format(min(adv_conv)))

    print('\nstandard statistic')
    print('num of samples: {}'.format(len(std_conv)))
    print('max of negative converage: {:.2f}'.format(max(std_conv)))
    print('mean of negative converage: {:.2f}'.format(sum(std_conv) / len(std_conv)))
    print('min of negative converage: {:.2f}'.format(min(std_conv)))


def main():
    opt = parser.parse_args()
    print(opt)
    guard_options(opt)

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    model = load_model(opt).to(device)

    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(_forward_conv(n))
    susp = load_select_info(opt)

    testset = load_dataset(opt)
    opt.num_test = len(testset)
    gene = GenericSearcher(opt, num_test=opt.num_test)

    for e in range(opt.fuzz_epoch):
        print('fuzz epoch =', e)
        mutators = gene.generate_next_population()
        testloader = generate_test_dataset(opt, testset, mutators)
        test(model, testloader, device, susp)
        gene.fitness()

    show_statisitc()
    print('[info] Done.')


if __name__ == "__main__":
    main()

