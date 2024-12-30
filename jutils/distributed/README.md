# Distributed PyTorch Training


In `min_DDP.py` you can find a minimum working example of single-node, multi-gpu training with PyTorch.
All communication between processes, as well as the multi-process spawn is handled by the functions defined in `distributed.py`.

```python
import torch
import torch.nn as nn
import jutils.distributed as dist

from torch.utils.data import DataLoader
```

### Main worker
First, you need to specify a main worker. This function is executed on every GPU individually.

```python
def main_worker(gpu, world_size):
    is_distributed = world_size > 1
    if is_distributed:
        dist.init_process_group(gpu, world_size)
    
    # you can either parse your arguments here or in the main function
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
```

### Training
Then you can specify the trainings loop.

```python
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

        # Up until now, all metrics are per gpu/process.  If
        # we want to get the metrics over all GPUs, we need to
        # communicate between processes. You can find a nice
        # visualization of communication schemes here:
        # https://pytorch.org/tutorials/intermediate/dist_tuto.html
        
        # synchronize metrics across gpus/processes
        loss = dist.reduce(loss.detach())               # average loss
        correct = dist.gather(correct.detach())         # gather all correct predictions
        correct = torch.cat(correct, dim=0)             # concatenate all correct predictions
        acc = correct.sum() / correct.numel()           # accuracy over all gpus/processes

        # metrics over all gpus, printed only in the main process
        if dist.is_primary():
            print(f"Finish iteration {it}"
                  f" - acc: {acc.cpu().item():.4f} ({correct.sum()}/{n})"
                  f" - loss: {loss.cpu().item():.4f}")
```

### Main
In the main function we only need to start the whole procedure.

```python
if __name__ == "__main__":
    dist.launch(main_worker)
```

### Usage

Run `min_DDP.py` with the following command on a multi-gpu machine
```
CUDA_VISIBLE_DEVICES="2,3" python3 min_DDP.py
```

The machine then only uses GPU 2 and 3 for training (attention: index starts at 0).

To run the example on a single, specific GPU, just enter
```
CUDA_VISIBLE_DEVICES="2" python3 min_DDP.py
```

In case the training gets interrupted without freeing the port, run
```
kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}')
```
to kill all `multiprocessing.spawn` related processes. 
