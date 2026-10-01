"""Deterministic dilation via interleaved batches; same Conv1d parameters.

Diagnostic candidate only: the submitted experiment does not import this file.
"""

import torch.nn.functional as F


def interleaved_conv1d(values, convolution):
    """Evaluate this experiment's same-length dilated convolution without dilation.

    Residue classes modulo dilation become independent batch items. Adjacent
    entries then represent positions separated by the original dilation.
    """
    dilation = convolution.dilation[0]
    if (convolution.kernel_size != (3,) or convolution.stride != (1,)
            or convolution.padding != (dilation,) or convolution.groups != 1
            or convolution.padding_mode != "zeros"):
        raise ValueError("Expected the experiment's width-three, same-length Conv1d")
    if dilation == 1:
        return convolution(values)
    batch, channels, length = values.shape
    padded = F.pad(values, (0, (-length) % dilation))
    grouped = padded.reshape(batch, channels, -1, dilation).permute(0, 3, 1, 2)
    grouped = grouped.reshape(batch * dilation, channels, -1)
    outputs = F.conv1d(grouped, convolution.weight, convolution.bias, padding=1)
    outputs = outputs.reshape(batch, dilation, convolution.out_channels, -1)
    return outputs.permute(0, 2, 3, 1).reshape(batch, convolution.out_channels, -1)[..., :length]


def interleaved_block_forward(self, hidden, mask):
    values = self.norm(hidden) * mask.unsqueeze(-1)
    values = interleaved_conv1d(values.transpose(1, 2), self.conv).transpose(1, 2)
    return (hidden + self.dropout(F.gelu(values))) * mask.unsqueeze(-1)
