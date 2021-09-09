import random
import itertools

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
        model, testloader, device, susp,
        desc="Evaluate", tqdm_leave=True):
    model.eval()

    global conv_info, adv_conv, std_conv

    result = []
    correct, total = 0, 0
    with tqdm(testloader, desc=desc, leave=tqdm_leave) as tepoch:
        for inputs, _, targets in tepoch:
            conv_info.clear()

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            acc = 100. * correct / total
            tepoch.set_postfix(acc=acc)

            # statistic converage
            for i, (p, t) in enumerate(zip(predicted, targets)):
                sum_negact, sum_negneu = 0, 0
                sum_neuact, sum_neuneu = 0, 0
                susp_l2chn = susp[str(t.item())]
                if any([ len(chns) == 0 for chns in susp_l2chn.values() ]):
                    continue
                for lname, chns in susp_l2chn.items():
                    act_info, num_neu = conv_info[lname]
                    sum_negact += act_info[i][chns].sum().item()
                    sum_negneu += (len(chns) * num_neu)
                for lname in conv_info.keys():
                    act_info, num_neu = conv_info[lname]
                    sum_neuact += act_info[i].sum().item()
                    sum_neuneu += (len(act_info[i]) * num_neu)
                negconv = sum_negact / sum_negneu if sum_negneu > 0 else 0
                neuconv = sum_neuact / sum_neuneu if sum_neuneu > 0 else 0
                result.append((negconv, neuconv, (p != t).int().item()))

    # sample balanced 1000 results for plotting
    num_to_sample = 1000
    acc = correct / total
    err_result = random.sample([r for r in result if r[2] == 1], int((1-acc) * num_to_sample))
    corr_result = random.sample([r for r in result if r[2] == 0], int(acc * num_to_sample))
    result = err_result + corr_result

    sorted_result = [r[2] for r in sorted(result, key=lambda x: x[0], reverse=True)]
    negconv_accu = list(itertools.accumulate(sorted_result))
    sorted_result = [r[2] for r in sorted(result, key=lambda x: x[1], reverse=True)]
    neuconv_accu = list(itertools.accumulate(sorted_result))
    random.shuffle(sorted_result)
    rand_accu = list(itertools.accumulate(sorted_result))

    return negconv_accu, neuconv_accu, rand_accu


def _forward_conv(lname):
    def __hook(module, finput, foutput):
        global conv_info
        b, c, *_ = foutput.size()
        squeeze = foutput.view(b, c, -1)
        num_neu = squeeze.size(-1)
        actives = (squeeze > 0).sum(dim=-1).cpu()
        conv_info[lname] = (actives, num_neu)
    return __hook


def export_echarts_option(opt, result, epoch):
    negconv_accu, neuconv_accu, rand_accu = result
    option = {
        'legend': {
            'data': ['random', 'neuron_conv', 'negative_conv']
        },
	'xAxis': {
	    'type': 'category',
            'boundaryGap': False,
	    'data': [str(i) for i in range(len(rand_accu))]
	},
	'yAxis': {
	    'type': 'value'
	},
	'series': [{
            'name': 'random',
            'data': rand_accu,
            'type': 'line'
        }, {
            'name': 'neuron_conv',
	    'data': neuconv_accu,
	    'type': 'line'
        }, {
            'name': 'negative_conv',
	    'data': negconv_accu,
	    'type': 'line'
	}]
    };
    export_object(opt, f'labeling_efforts_e{epoch}.json', option, indent=None)


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

    collect_epoch = [1, 5, 10, 20]
    for e in range(opt.fuzz_epoch):
        print('fuzz epoch =', e)
        mutators = gene.generate_next_population()
        testloader = generate_test_dataset(opt, testset, mutators)
        result = test(model, testloader, device, susp)
        if (e + 1) in collect_epoch:
            export_echarts_option(opt, result, e+1)
        gene.fitness()

    print('[info] Done.')


if __name__ == "__main__":
    main()

