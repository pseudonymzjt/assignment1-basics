import numpy as np
import torch


def get_batch(x, batch_size, context_length, device):
    """
    Args:
        x: numpy array of token IDs, shape (num_tokens,)
        batch_size: 批次大小
        context_length: 上下文窗口长度
        device: PyTorch设备 ('cpu' 或 'cuda:0')
    
    Returns:
        inputs: shape (batch_size, context_length)
        targets: shape (batch_size, context_length)
    """
    # initial a legal index
    max_start_idx = len(x) - context_length
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    
    # build input and targets
    inputs = np.array([x[i:i+context_length] for i in start_indices])
    targets = np.array([x[i+1:i+context_length+1] for i in start_indices])
    
    # transfer into pytorch tensor for specified device
    inputs = torch.from_numpy(inputs).long().to(device)
    targets = torch.from_numpy(targets).long().to(device)
    
    return inputs, targets

def save_checkpoint(model, optimizer, iteration, out):
    '''
    save_checkpoint should dump all the state from the model, optimizer and iteration into the file-like object out. 
    You can use the state_dict method of both the model and the optimizer to get their relevant states and use 
    torch.save(obj, out) to dump obj into out (PyTorch supports either a path or a file-like 
    object here). 
    A typical choice is to have obj be a dictionary, but you can use whatever format you want 
    as long as you can load your checkpoint later.
    This function expects the following parameters:
    model: torch.nn.Module  
    optimizer: torch.optim.Optimizer  
    iteration: int  
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    '''
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    '''
    should load a checkpoint from src (path or file-like 
    object), and then recover the model and optimizer states from that checkpoint. Your function 
    should return the iteration number that was saved to the checkpoint. You can use 
    torch.load(src) to recover what you saved in your save_checkpoint implementation, and the 
    load_state_dict method in both the model and optimizer to return them to their previous 
    states.
    This function expects the following parameters:
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]  
    model: torch.nn.Module  
    optimizer: torch.optim.Optimizer
    '''
    checkpoint = torch.load(src)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    iteration = checkpoint['iteration']
    
    return iteration