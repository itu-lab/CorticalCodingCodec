import torch
import torch.nn.functional as F

def haar_wavelet_decomposition(data, levels:int=None):
    """
    Perform multi-level Haar wavelet decomposition on a batch of signals.

    Args:
    data (Tensor): The input tensor of shape (batch_size, signal_length).
    levels (int): Number of decomposition levels.

    Returns:
    Tensor: The decomposed signals.
    """
    if levels is None:
        levels = int(np.log2(data.shape[1]))

    batch_size, signal_length = data.shape

    # Haar filters
    low_pass_filter = torch.tensor([0.7071, 0.7071]).view(1, 1, 2).float().to(data.device)
    high_pass_filter = torch.tensor([0.7071, -0.7071]).view(1, 1, 2).float().to(data.device)

    batch_stack = 1
    for _ in range(levels):
        # Apply the filters
        approx = F.conv1d(data.view(batch_size*batch_stack, 1, -1), low_pass_filter, stride=2)
        detail = F.conv1d(data.view(batch_size*batch_stack, 1, -1), high_pass_filter, stride=2)

        # Concatenate approximations and details
        data = torch.cat((approx, detail), dim=1)  
        batch_stack *= 2

    # reindex the data 
    data = data.view(batch_size, -1)
    
    return data

def haar_wavelet_reconstruction(data, levels:int=None):
    """
    Perform multi-level inverse Haar wavelet reconstruction on decomposed signals.

    Args:
    data (Tensor): The input tensor containing the decomposed signals.
    levels (int): Number of decomposition levels.

    Returns:
    Tensor: The reconstructed original signals.
    """
    if levels is None:
        levels = int(np.log2(data.shape[1]))

    batch_size, signal_length = data.shape

    # Haar filters
    low_pass_filter = torch.tensor([0.7071, 0.7071]).view(1, 1, 2).float().to(data.device)
    high_pass_filter = torch.tensor([0.7071, -0.7071]).view(1, 1, 2).float().to(data.device)

    current_length = signal_length // (2 ** levels)  # Initial length of the approximation and detail components

    for _ in range(levels):
        data = data.view(-1, 2, current_length)
        d_approx, d_detail = data.unbind(dim=1)
        d_approx = d_approx.view(-1, 1, current_length)
        d_detail = d_detail.view(-1, 1, current_length)
        data = F.conv_transpose1d(d_approx, low_pass_filter, stride=2) + F.conv_transpose1d(d_detail, high_pass_filter, stride=2)
        current_length *= 2

    data = data.view(-1, 8)

    return data