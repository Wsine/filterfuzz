import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, kendalltau

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

    global conv_info

    result = []
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

            # statistic converage
            for i, (p, t) in enumerate(zip(predicted, targets)):
                susp_l2chn = susp[str(t.item())]
                if any([ len(chns) == 0 for chns in susp_l2chn.values() ]):
                    continue

                sum_negact, sum_negneu = 0, 0
                for lname, chns in susp_l2chn.items():
                    act_info, num_neu = conv_info[lname]
                    sum_negact += act_info[i][chns].sum().item()
                    sum_negneu += (len(chns) * num_neu)
                negconv = sum_negact / sum_negneu if sum_negneu > 0 else 0

                sum_neuact, sum_neuneu = 0, 0
                for lname in conv_info.keys():
                    act_info, num_neu = conv_info[lname]
                    sum_neuact += act_info[i].sum().item()
                    sum_neuneu += (len(act_info[i]) * num_neu)
                neuconv = sum_neuact / sum_neuneu if sum_neuneu > 0 else 0

                result.append((negconv, neuconv, (p == t).int().item()))

    return result


def _forward_conv(lname):
    def __hook(module, finput, foutput):
        global conv_info
        b, c, *_ = foutput.size()
        squeeze = foutput.view(b, c, -1)
        num_neu = squeeze.size(-1)
        actives = (squeeze > 0).sum(dim=-1).cpu()
        conv_info[lname] = (actives, num_neu)
    return __hook


def show_statisitc(result):
    print('adversarial statistic')
    adv_conv = [r[0] for r in result if r[2] == 0]
    print('num of samples: {}'.format(len(adv_conv)))
    print('max of negative converage: {:.2f}'.format(max(adv_conv)))
    print('mean of negative converage: {:.2f}'.format(sum(adv_conv) / len(adv_conv)))
    print('min of negative converage: {:.2f}'.format(min(adv_conv)))

    print('\nstandard statistic')
    std_conv = [r[0] for r in result if r[2] == 1]
    print('num of samples: {}'.format(len(std_conv)))
    print('max of negative converage: {:.2f}'.format(max(std_conv)))
    print('mean of negative converage: {:.2f}'.format(sum(std_conv) / len(std_conv)))
    print('min of negative converage: {:.2f}'.format(min(std_conv)))

    print('\ncorrcoef statistic')
    negconv = [r[0] for r in result]
    neuconv = [r[1] for r in result]
    predict = [r[2] for r in result]
    print("negconv's pearson:", pearsonr(negconv, predict))
    print("negconv's spearman:", spearmanr(negconv, predict))
    print("negconv's kendall:", kendalltau(negconv, predict))
    print("neuconv's pearson:", pearsonr(neuconv, predict))
    print("neuconv's spearman:", spearmanr(neuconv, predict))
    print("neuconv's kendall:", kendalltau(neuconv, predict))

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
        result = test(model, testloader, device, susp)
        show_statisitc(result)
        gene.fitness()

    print('[info] Done.')


if __name__ == "__main__":
    main()

