import torch
import torch.nn.functional as F

def compute_strides(shape):
    strides = [1]
    for dim in reversed(shape[1:]):
        strides.append(strides[-1] * dim)
    return list(reversed(strides))

def aggregate_with_counts(tensor, count_tensor, idx, data, weight=None):
    dims = idx.shape[-1]
    channels = tensor.shape[-1]
    spatial_dims = tensor.shape[:-1]
    strides = compute_strides(spatial_dims)

    assert len(spatial_dims) == dims

    tensor = tensor.view(-1, channels)
    count_tensor = count_tensor.view(-1, 1)

    flat_idx = (idx * torch.tensor(strides, dtype=torch.long, device=idx.device)).sum(-1)

    if weight is None:
        weight = torch.ones_like(data[..., :1])

    tensor.scatter_add_(0, flat_idx.unsqueeze(1).repeat(1, channels), data)
    count_tensor.scatter_add_(0, flat_idx.unsqueeze(1), weight)

    return tensor.view(*spatial_dims, channels), count_tensor.view(*spatial_dims, 1)

def nearest_insert_2d(height, width, positions, data, return_counts=False):
    channels = data.shape[-1]

    idx = (positions * 0.5 + 0.5) * torch.tensor(
        [height - 1, width - 1], dtype=torch.float32, device=positions.device
    )
    idx = idx.round().long()

    output = torch.zeros(height, width, channels, device=data.device, dtype=data.dtype)
    counts = torch.zeros(height, width, 1, device=data.device, dtype=data.dtype)
    weight = torch.ones_like(data[..., :1])

    output, counts = aggregate_with_counts(output, counts, idx, data, weight)

    if return_counts:
        return output, counts

    mask = (counts.squeeze(-1) > 0)
    output[mask] = output[mask] / counts[mask].repeat(1, channels)

    return output

def bilinear_insert_2d(height, width, positions, data, return_counts=False):
    channels = data.shape[-1]

    idx = (positions * 0.5 + 0.5) * torch.tensor(
        [height - 1, width - 1], dtype=torch.float32, device=positions.device
    )
    base_idx = idx.floor().long()
    base_idx[:, 0].clamp_(0, height - 2)
    base_idx[:, 1].clamp_(0, width - 2)

    idx_offsets = [
        torch.tensor([0, 0], dtype=torch.long, device=positions.device),
        torch.tensor([0, 1], dtype=torch.long, device=positions.device),
        torch.tensor([1, 0], dtype=torch.long, device=positions.device),
        torch.tensor([1, 1], dtype=torch.long, device=positions.device),
    ]
    neighbor_idxs = [base_idx + offset for offset in idx_offsets]

    h = idx[..., 0] - base_idx[..., 0].float()
    w = idx[..., 1] - base_idx[..., 1].float()
    weights = [
        (1 - h) * (1 - w),
        (1 - h) * w,
        h * (1 - w),
        h * w,
    ]

    output = torch.zeros(height, width, channels, device=data.device, dtype=data.dtype)
    counts = torch.zeros(height, width, 1, device=data.device, dtype=data.dtype)
    base_weight = torch.ones_like(data[..., :1])

    for neighbor_idx, weight in zip(neighbor_idxs, weights):
        weighted_data = data * weight.unsqueeze(1)
        weighted_weight = base_weight * weight.unsqueeze(1)
        output, counts = aggregate_with_counts(output, counts, neighbor_idx, weighted_data, weighted_weight)

    if return_counts:
        return output, counts

    mask = (counts.squeeze(-1) > 0)
    output[mask] = output[mask] / counts[mask].repeat(1, channels)

    return output

def mipmap_bilinear_insert_2d(height, width, positions, data, min_res=32, return_counts=False):
    channels = data.shape[-1]

    output = torch.zeros(height, width, channels, device=data.device, dtype=data.dtype)
    counts = torch.zeros(height, width, 1, device=data.device, dtype=data.dtype)

    curr_height, curr_width = height, width

    while min(curr_height, curr_width) > min_res:
        mask = (counts.squeeze(-1) == 0)
        if not mask.any():
            break

        curr_output, curr_counts = bilinear_insert_2d(curr_height, curr_width, positions, data, return_counts=True)
        interpolated_output = F.interpolate(curr_output.permute(2, 0, 1).unsqueeze(0), (height, width), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
        interpolated_counts = F.interpolate(curr_counts.permute(2, 0, 1).unsqueeze(0), (height, width), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)

        output[mask] += interpolated_output[mask]
        counts[mask] += interpolated_counts[mask]

        curr_height //= 2
        curr_width //= 2

    if return_counts:
        return output, counts

    mask = (counts.squeeze(-1) > 0)
    output[mask] = output[mask] / counts[mask].repeat(1, channels)

    return output

def nearest_insert_3d(height, width, depth, positions, data, return_counts=False):
    channels = data.shape[-1]

    idx = (positions * 0.5 + 0.5) * torch.tensor(
        [height - 1, width - 1, depth - 1], dtype=torch.float32, device=positions.device
    )
    idx = idx.round().long()

    output = torch.zeros(height, width, depth, channels, device=data.device, dtype=data.dtype)
    counts = torch.zeros(height, width, depth, 1, device=data.device, dtype=data.dtype)
    weight = torch.ones_like(data[..., :1])

    output, counts = aggregate_with_counts(output, counts, idx, data, weight)

    if return_counts:
        return output, counts

    mask = (counts.squeeze(-1) > 0)
    output[mask] = output[mask] / counts[mask].repeat(1, channels)

    return output

def trilinear_insert_3d(height, width, depth, positions, data, return_counts=False):
    channels = data.shape[-1]

    idx = (positions * 0.5 + 0.5) * torch.tensor(
        [height - 1, width - 1, depth - 1], dtype=torch.float32, device=positions.device
    )
    base_idx = idx.floor().long()
    base_idx[:, 0].clamp_(0, height - 2)
    base_idx[:, 1].clamp_(0, width - 2)
    base_idx[:, 2].clamp_(0, depth - 2)

    idx_offsets = [
        torch.tensor([0, 0, 0], dtype=torch.long, device=positions.device),
        torch.tensor([0, 0, 1], dtype=torch.long, device=positions.device),
        torch.tensor([0, 1, 0], dtype=torch.long, device=positions.device),
        torch.tensor([0, 1, 1], dtype=torch.long, device=positions.device),
        torch.tensor([1, 0, 0], dtype=torch.long, device=positions.device),
        torch.tensor([1, 0, 1], dtype=torch.long, device=positions.device),
        torch.tensor([1, 1, 0], dtype=torch.long, device=positions.device),
        torch.tensor([1, 1, 1], dtype=torch.long, device=positions.device),
    ]
    neighbor_idxs = [base_idx + offset for offset in idx_offsets]

    h = idx[..., 0] - base_idx[..., 0].float()
    w = idx[..., 1] - base_idx[..., 1].float()
    d = idx[..., 2] - base_idx[..., 2].float()
    weights = [
        (1 - h) * (1 - w) * (1 - d),
        (1 - h) * (1 - w) * d,
        (1 - h) * w * (1 - d),
        (1 - h) * w * d,
        h * (1 - w) * (1 - d),
        h * (1 - w) * d,
        h * w * (1 - d),
        h * w * d,
    ]

    output = torch.zeros(height, width, depth, channels, device=data.device, dtype=data.dtype)
    counts = torch.zeros(height, width, depth, 1, device=data.device, dtype=data.dtype)
    base_weight = torch.ones_like(data[..., :1])

    for neighbor_idx, weight in zip(neighbor_idxs, weights):
        weighted_data = data * weight.unsqueeze(1)
        weighted_weight = base_weight * weight.unsqueeze(1)
        output, counts = aggregate_with_counts(output, counts, neighbor_idx, weighted_data, weighted_weight)

    if return_counts:
        return output, counts

    mask = (counts.squeeze(-1) > 0)
    output[mask] = output[mask] / counts[mask].repeat(1, channels)

    return output

def mipmap_trilinear_insert_3d(height, width, depth, positions, data, min_res=32, return_counts=False):
    # positions: [N, 3], float in [-1, 1]
    # data: [N, C]

    channels = data.shape[-1]
    out_tensor = torch.zeros(height, width, depth, channels, device=data.device, dtype=data.dtype)
    counter = torch.zeros(height, width, depth, 1, device=data.device, dtype=data.dtype)

    curr_h, curr_w, curr_d = height, width, depth

    while min(curr_h, curr_w, curr_d) > min_res:
        mask = (counter.squeeze(-1) == 0)
        if not mask.any():
            break

        partial_tensor, partial_count = trilinear_insert_3d(curr_h, curr_w, curr_d, positions, data, return_counts=True)

        resized_tensor = F.interpolate(
            partial_tensor.permute(3, 0, 1, 2).unsqueeze(0),
            size=(height, width, depth),
            mode='trilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 3, 0).contiguous()

        resized_count = F.interpolate(
            partial_count.permute(3, 0, 1, 2).unsqueeze(0),
            size=(height, width, depth),
            mode='trilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 3, 0).contiguous()

        out_tensor[mask] += resized_tensor[mask]
        counter[mask] += resized_count[mask]

        curr_h //= 2
        curr_w //= 2
        curr_d //= 2

    if return_counts:
        return out_tensor, counter

    valid_mask = (counter.squeeze(-1) > 0)
    out_tensor[valid_mask] = out_tensor[valid_mask] / counter[valid_mask].repeat(1, channels)

    return out_tensor

def insert_in_grid(
    positions,
    data,
    shape,
    mode='bilinear',  # Options: 'nearest', 'bilinear', 'mipmap'
    return_counts=False
):
    dim = positions.shape[-1]
    assert dim in [2, 3], "Only 2D and 3D insertions are supported"

    if dim == 2:
        height, width = shape
        if mode == 'nearest':
            return nearest_insert_2d(height, width, positions, data, return_counts)
        elif mode == 'bilinear':
            return bilinear_insert_2d(height, width, positions, data, return_counts)
        elif mode == 'mipmap':
            return mipmap_bilinear_insert_2d(height, width, positions, data, return_counts)
        else:
            raise ValueError(f"Unsupported mode for 2D: {mode}")

    elif dim == 3:
        height, width, depth = shape
        if mode == 'nearest':
            return nearest_insert_3d(height, width, depth, positions, data, return_counts)
        elif mode == 'bilinear':
            return trilinear_insert_3d(height, width, depth, positions, data, return_counts)
        elif mode == 'mipmap':
            return mipmap_trilinear_insert_3d(height, width, depth, positions, data, return_counts)
        else:
            raise ValueError(f"Unsupported mode for 3D: {mode}")
