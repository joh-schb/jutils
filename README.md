# jutils

Some useful utility functions.

Simply install it with

```
pip install git+https://github.com/joh-fischer/jutils.git#egg=jutils
```

## Usage

Please check the `test_all.py` file or the individual python files for usage examples. For example, you
can find functionality for vision in the `jutils/vision` folder, which include depth map colorization,
as well as image and video processing.


## Checkpoints

Pre-trained pytorch checkpoints for the models can be downloaded like this:

```
mkdir checkpoints
cd checkpoints

# SD Autoencoder checkpoint
wget -O sd_ae.ckpt https://www.dropbox.com/scl/fi/lvfvy7qou05kxfbqz5d42/sd_ae.ckpt?rlkey=fvtu2o48namouu9x3w08olv3o&st=vahu44z5&dl=0

# TinyAutoencoderKL checkpoints
wget https://github.com/madebyollin/taesd/raw/refs/heads/main/taesd_encoder.pth
wget https://github.com/madebyollin/taesd/raw/refs/heads/main/taesd_decoder.pth
```


## Pytorch Distributed

In `jutils/distributed/min_DDP.py` you find a minimal working example of how to properly do multi-GPU training in PyTorch. All communication between processes, as well as the multi-process spawn is handled by the functions defined in `distributed.py`.

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

    # sync all processes
    dist.barrier()

    # kill process group
    dist.cleanup()
```

### Training
Then you can specify the training-loop.

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
            print(f"iteration {it} | acc: {acc.cpu().item():.4f}, loss: {loss.cpu().item():.4f}")

        # for printing on rank 0 you could also use
        dist.print0('Only on rank 0')
```

### Main
In the main function we only need to start the whole procedure.

```python
if __name__ == "__main__":
    dist.launch(main_worker)
```
