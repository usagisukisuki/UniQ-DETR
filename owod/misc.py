import torch.distributed as dist


def all_gather_object(obj):
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out
