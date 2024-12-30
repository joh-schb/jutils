import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import jutils.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Multi-GPU Training')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='Number of training epochs.')
    parser.add_argument('--batch-size', default=8, type=int, metavar='N',
                        help='Batch size.')
    # data
    parser.add_argument('--n-classes', default=4, type=int, metavar='N',
                        help='Number of classes for fake dataset.')
    parser.add_argument('--data-size', default=32, type=int, metavar='N',
                        help='Size of fake dataset.')
    parser.add_argument('--hidden-dim', default=32, type=int, metavar='N',
                        help='Hidden dimension.')
    args = parser.parse_args()
    return args


class DummyDataset(Dataset):
    def __init__(self, length, n_classes):
        self.len = length
        gen = torch.Generator().manual_seed(0)
        self.data = torch.arange(0, length, dtype=torch.float32).unsqueeze(-1)
        self.labels = torch.randint(0, n_classes, (length,), generator=gen)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return self.len


class DummyModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = self.lin2(self.lin1(x))
        return x


# Main workers ##################
def main_worker(gpu, world_size):
    is_distributed = world_size > 1
    if is_distributed:
        dist.init_process_group(gpu, world_size)

    args = parse_args()
    for name, val in vars(args).items():
        dist.print_primary("{:<12}: {}".format(name, val))

    """ Data """
    dataset = DummyDataset(args.data_size, args.n_classes)
    sampler = dist.data_sampler(dataset, is_distributed, shuffle=False)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=(sampler is None), sampler=sampler)

    """ Model """
    model = DummyModel(in_dim=1, hidden_dim=args.hidden_dim, n_classes=args.n_classes)
    model.to(dist.get_device())
    model = dist.prepare_ddp_model(model, device_ids=[gpu])

    """ Optimizer and Loss """
    optimizer = torch.optim.AdamW(model.parameters(), 0.0001)
    criterion = nn.CrossEntropyLoss()

    """ Run Epochs """
    print("Run epochs")
    for epoch in range(args.epochs):
        dist.print_primary(f"------- Epoch {epoch+1}")
        
        if is_distributed:
            sampler.set_epoch(epoch)

        # training
        train(model, loader, criterion, optimizer)

    # kill process group
    dist.cleanup()


def train(model, loader, criterion, optimizer):
    model.train()

    for it, (x, y) in enumerate(loader):
        x, y = x.to(dist.get_device()), y.to(dist.get_device())

        y_hat = model(x)
    
        loss = criterion(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct = torch.argmax(y_hat, dim=1).eq(y)
        n = y.shape[0]

        # metrics per gpu/process
        print(f"Device: {x.device}"
              f"\n\tInput: \t{x.squeeze().to(torch.uint8)}"
              f"\n\tLabel: \t{y.squeeze()}"
              f"\n\tPred:  \t{torch.argmax(y_hat, -1)}"
              f"\n\tCorr.: \t{correct.to(torch.uint8)}"
              f"\n\tAcc:   \t{correct.sum() / n:.5f} ({correct.sum()}/{n})"
              f"\n\tLoss:  \t{loss.cpu().item():.5f}")

        # wait until all processes are at this point
        dist.wait_for_everyone()

        # synchronize metrics across gpus/processes
        loss = dist.reduce(loss.detach())               # average loss
        correct = dist.gather(correct.detach())         # gather all correct predictions
        correct = torch.cat(correct, dim=0)             # concatenate all correct predictions
        acc = correct.sum() / correct.numel()           # accuracy over all gpus/processes

        # metrics over all gpus, printed only on the main process
        dist.print_primary(f"Finish iteration {it}"
                           f" - acc: {acc.cpu().item():.4f} ({correct.sum()}/{correct.shape[0]})"
                           f" - loss: {loss.cpu().item():.4f}")


if __name__ == "__main__":
    # code that should only execute once
    # ...

    # start different processes, if multi-gpu is available
    # otherwise, it just starts the main_worker once
    dist.launch(main_worker)
