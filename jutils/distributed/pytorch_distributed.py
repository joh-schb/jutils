# MIT License Copyright (c) 2022 joh-schb
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import torch
import socket
from contextlib import closing

import torch.multiprocessing as mp
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


# launch
def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = s.getsockname()[1]
        return port


def launch(worker_fn, *args):
    world_size = torch.cuda.device_count()

    if world_size > 1:          # distributed training
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            raise ValueError("GPUs not specified. Please set CUDA_VISIBLE_DEVICES.")

        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(find_free_port())

        mp.spawn(worker_fn, args=(world_size, *args),
                 nprocs=world_size, join=True)

    elif world_size == 1:       # single GPU training
        worker_fn(0, world_size, *args)

    else:                       # CPU training
        worker_fn(0, world_size, *args)


# distributed trainings functions
def init_process_group(rank, world_size, backend=None):
    if backend is None:
        backend = "gloo" if not torch.cuda.is_available() else "nccl"
    dist.init_process_group(backend, init_method="env://",
                            rank=rank, world_size=world_size)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def cleanup():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{get_rank()}")
    return torch.device("cpu")


def is_primary():
    return get_rank() == 0


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


# data loading stuff
def data_sampler(dataset, distributed, shuffle):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)
    return None


# model wrapping
def prepare_ddp_model(model, device_ids, *args, **kwargs):
    if get_world_size() > 1:
        model = DistributedDataParallel(model, device_ids=device_ids, *args, **kwargs)
    return model


# synchronization functions
def all_reduce(tensor, op='sum'):
    world_size = get_world_size()

    if world_size == 1:
        return tensor

    if op == 'sum':
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    elif op == 'avg':
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= get_world_size()
    else:
        raise ValueError(f'"{op}" is an invalid reduce operation!')

    return tensor


def reduce(tensor, op=dist.ReduceOp.SUM):
    world_size = get_world_size()

    if world_size == 1:
        return tensor

    dist.reduce(tensor, dst=0, op=op)

    return tensor


def gather(data):
    world_size = get_world_size()

    if world_size == 1:
        return [data]

    output_list = [torch.zeros_like(data) for _ in range(world_size)]

    if is_primary():
        dist.gather(data, gather_list=output_list)
    else:
        dist.gather(data)

    return output_list


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    if is_dist_avail_and_initialized():
        for p in params:
            with torch.no_grad():
                dist.broadcast(p, 0)


def barrier():
    world_size = get_world_size()
    if world_size == 1:
        return
    dist.barrier()


# wrapper with same functionality but better readability as barrier
def wait_for_everyone():
    barrier()


def print_primary(*args, **kwargs):
    if is_primary():
        print(*args, **kwargs)


def print0(*args, **kwargs):
    print_primary(*args, **kwargs)
