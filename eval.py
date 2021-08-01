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
        for inputs, targets in tepoch:
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


def main():
    opt = parser.parse_args()
    print(opt)
    guard_folder(opt)

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    model = load_model(opt).to(device)

    testset = load_dataset(opt)
    testloader = generate_test_dataset(opt, testset)

    num_classes = Args.get_num_class(opt.dataset)
    base_acc, base_c_acc = test(model, testloader, device, num_classes)
    print('base_acc: {:.2f}%'.format(base_acc))
    print('base_class_acc:', base_c_acc)

if __name__ == "__main__":
    main()

