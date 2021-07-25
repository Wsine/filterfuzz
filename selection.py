import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import generate_test_dataset, load_dataset
from generic_search import GenericSearcher
from model import load_model
from arguments import Args, parser
from utils import *


@torch.no_grad()
def test(model, testloader, device, num_classes, desc="Evaluate", tqdm_leave=True):
    model.eval()
    confusion_matrix = torch.zeros(num_classes, num_classes)
    correct, total = 0, 0
    with tqdm(testloader, desc=desc, leave=tqdm_leave) as tepoch:
        for inputs, _, targets in tepoch:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            acc = 100. * correct / total
            tepoch.set_postfix(acc=acc)

            for t, p in zip(targets, predicted):
                confusion_matrix[t.item(), p.item()] += 1

    acc = 100. * correct / total
    class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
    return acc, class_acc


def performance_loss(opt, model, valloader, device):
    num_classes = Args.get_num_class(opt.dataset)
    base_acc, base_c_acc = test(model, valloader, device, num_classes)
    print('base_acc =', base_acc)
    print('base_class_acc =', base_c_acc)

    def _mask_out_channel(chn):
        def __hook(module, finput, foutput):
            foutput[:, chn] = 0
            return foutput
        return __hook

    suspicious = { str(c): {} for c in range(num_classes) }
    conv_names = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    for lname in tqdm(conv_names, desc="Modules"):
        module = rgetattr(model, lname)
        indices = [[] for _ in range(num_classes)]
        for chn in tqdm(range(module.out_channels), desc="Filters", leave=False):
            handle = module.register_forward_hook(_mask_out_channel(chn))
            _, c_acc = test(model, valloader, device, num_classes, tqdm_leave=False)
            for c in range(num_classes):
                if c_acc[c] > base_c_acc[c]:
                    indices[c].append(chn)
            handle.remove()
        for c in range(num_classes):
            suspicious[str(c)][lname] = indices[c]

    #  l2module = rgetattr(model, 'fc1')
    #  num_neurons = l2module.out_features
    #  suspicious = { str(c): [] for c in range(num_classes) }
    #  for nidx in tqdm(range(num_neurons), desc="Neurons"):
    #      handle = l2module.register_forward_hook(_mask_out_channel(nidx))
    #      _, c_acc = test(model, valloader, device, num_classes, tqdm_leave=False)
    #      for c in range(num_classes):
    #          if c_acc[c] > base_c_acc[c]:
    #              suspicious[str(c)].append(nidx)
    #      handle.remove()

    return suspicious



def main():
    opt = parser.parse_args()
    print(opt)
    guard_folder(opt)

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    model = load_model(opt).to(device)

    valset = load_dataset(opt, set_type='val')
    gene = GenericSearcher(opt, num_test=len(valset))
    mutators = gene.generate_next_population()
    valloader = generate_test_dataset(opt, valset, mutators)

    result = performance_loss(opt, model, valloader, device)
    result_name = "susp_filters.json"
    export_object(opt, result_name, result)


if __name__ == "__main__":
    main()

