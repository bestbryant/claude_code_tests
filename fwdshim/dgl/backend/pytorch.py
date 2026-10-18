"""dgl.backend.pytorch: packed <-> padded conversions (rows of variable length)."""
import torch


def pad_packed_tensor(input, lengths, value, l_min=None):
    lengths = torch.as_tensor(lengths, dtype=torch.long, device=input.device)
    max_len = int(lengths.max().item()) if lengths.numel() else 0
    if l_min is not None:
        max_len = max(max_len, int(l_min))
    out = input.new_full((lengths.numel(), max_len) + tuple(input.shape[1:]), value)
    row = torch.repeat_interleave(torch.arange(lengths.numel(), device=input.device), lengths)
    start = torch.cumsum(lengths, 0) - lengths
    col = torch.arange(input.shape[0], device=input.device) - torch.repeat_interleave(start, lengths)
    out[row, col] = input
    return out


def pack_padded_tensor(input, lengths):
    lengths = torch.as_tensor(lengths, dtype=torch.long, device=input.device)
    mask = torch.arange(input.shape[1], device=input.device).unsqueeze(0) < lengths.unsqueeze(1)
    return input[mask]
