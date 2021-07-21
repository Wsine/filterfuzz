import logging

import torch
from tqdm import tqdm

from dataset import generate_test_dataset, load_dataset
from generic_search import GenericSearcher
from model import load_model
from arguments import parser
from utils import *


@torch.no_grad()
def test(model, testloader, device, desc="Evaluate", tqdm_leave=True):
    model.eval()
    correct, total = 0, 0
    with tqdm(testloader, desc=desc, leave=tqdm_leave) as tepoch:
        for inputs, descs, targets in tepoch:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            total += targets.size(0)
            _, predicted = outputs.max(1)
            comp = predicted.eq(targets)
            correct += comp.sum().item()

            acc = 100. * correct / total
            tepoch.set_postfix(acc=acc)

            err_idx = (~comp).nonzero().flatten().tolist()
            for eid in err_idx:
                p = predicted[eid].item()
                t = targets[eid].item()
                logging.info('%s | %s', descs[eid], f'predict: {p}, target: {t}')

    acc = 100. * correct / total
    return acc


def main():
    opt = parser.parse_args()
    print(opt)
    guard_folder(opt)

    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        filename=os.path.join(opt.output_dir, opt.dataset, opt.model, 'adversarial_samples.log'),
        filemode='w',
        level=logging.INFO
    )

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    model = load_model(opt).to(device)

    testset = load_dataset(opt)
    gene = GenericSearcher(opt, num_test=len(testset))

    mutators = gene.generate_next_population()
    testloader = generate_test_dataset(opt, testset, mutators)
    #  testloader = generate_test_dataset(opt, testset)
    acc = test(model, testloader, device)
    print('test accuracy is {:.4f}'.format(acc))


if __name__ == "__main__":
    main()

