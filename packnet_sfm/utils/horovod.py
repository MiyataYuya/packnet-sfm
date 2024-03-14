HAS_HOROVOD = False


def hvd_init():
    return HAS_HOROVOD

def on_rank_0(func):
    def wrapper(*args, **kwargs):
        if rank() == 0:
            func(*args, **kwargs)
    return wrapper

def rank():
    return 0

def world_size():
    return 1

@on_rank_0
def print0(string='\n'):
    print(string)

def reduce_value(value, average, name):
    """
    Reduce the mean value of a tensor from all GPUs

    Parameters
    ----------
    value : torch.Tensor
        Value to be reduced
    average : bool
        Whether values will be averaged or not
    name : str
        Value name

    Returns
    -------
    value : torch.Tensor
        reduced value
    """
    # print("average: ", average, "name: ", name)
    # return hvd.allreduce(value, average=average, name=name)
    return value
