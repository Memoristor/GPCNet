# coding=utf-8

__all__ = [
    "carafe_flop_jit",
    "sdpa_flop_jitter",
    "selective_scan_flop_jit",
    "mamba_inner_fn_no_out_proj_flop_jit",
]


def sdpa_flop_jitter(inputs, outputs):
    """
    Flop jitter for Scaled Dot-Product Attention (SDPA).
    """
    # query shape: [B, H, S_q, D]
    # key shape:   [B, H, S_kv, D]
    query = inputs[0]
    key = inputs[1]

    q_shape = query.type().sizes()
    k_shape = key.type().sizes()

    batch = q_shape[0]
    heads = q_shape[1]
    seq_len_q = q_shape[2]
    seq_len_kv = k_shape[2]
    dim_head = q_shape[3]

    # 4 * B * H * Sq * Skv * D
    total_flops = 4 * batch * heads * seq_len_q * seq_len_kv * dim_head
    return total_flops


def infer_carafe_kernel_size_and_groups(mask_channels, feature_channels):
    """Infer CARAFE kernel size and groups from mask channels."""
    candidates = []
    for k in [1, 3, 5, 7, 9, 11]:
        k2 = k * k
        if mask_channels % k2 != 0:
            continue
        groups = mask_channels // k2
        if feature_channels % groups != 0:
            continue
        candidates.append((k, groups))

    if len(candidates) == 0:
        raise ValueError(
            f"Cannot infer CARAFE kernel_size: mask_channels={mask_channels}, "
            f"feature_channels={feature_channels}, no valid (kernel_size, groups) found."
        )
    if len(candidates) > 1:
        candidates = sorted(candidates, key=lambda item: item[0], reverse=True)
    return candidates[0]


def carafe_flop_jit(inputs, outputs):
    """Count FLOPs for mmcv.ops.carafe CARAFEFunction.

    The PythonOp exposes only features and masks to fvcore. The reassembly
    kernel size is inferred from mask channels: C_mask = kernel_size^2 * groups.
    FLOPs are counted as multiply-adds, matching the convention used by
    ``svc_flop_jit`` in this file.
    """
    features = inputs[0]
    masks = inputs[1]
    output = outputs[0]

    feature_shape = features.type().sizes()
    mask_shape = masks.type().sizes()
    output_shape = output.type().sizes()

    batch_size = output_shape[0]
    channels = output_shape[1]
    out_h = output_shape[2]
    out_w = output_shape[3]

    feature_channels = feature_shape[1]
    mask_channels = mask_shape[1]
    kernel_size, _ = infer_carafe_kernel_size_and_groups(mask_channels, feature_channels)

    num_output_elements = batch_size * channels * out_h * out_w
    flops_per_element = kernel_size * kernel_size * 2
    return num_output_elements * flops_per_element


def infer_kernel_size_and_groups(weight_channels, in_channels):
    """Infer kernel size and groups for SpatialVariableConvolution based on weight channels and input channels."""
    candidates = []
    for k in [1, 3, 5, 7]:
        k2 = k * k
        if weight_channels % k2 != 0:
            continue
        groups = weight_channels // k2
        if in_channels % groups != 0:
            continue
        candidates.append((k, groups))

    if len(candidates) == 0:
        raise ValueError(
            f"Cannot infer kernel_size: weight_channels={weight_channels}, "
            f"in_channels={in_channels}, no valid (k, groups) found."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous kernel_size inference: candidates={candidates}, "
            f"weight_channels={weight_channels}, in_channels={in_channels}."
        )
    return candidates[0]  # (kernel_size, groups)


def svc_flop_jit(inputs, outputs):
    """Count FLOPs for SpatialVariableConvolutionFunction."""
    input_tensor = inputs[0]
    weight = inputs[1]
    dilation_map = None
    if len(inputs) == 3:
        dilation_map = inputs[2]

    outputs = outputs[0]

    input_shape = input_tensor.type().sizes()
    batch_size = input_shape[0]
    channels = input_shape[1]

    weight_shape = weight.type().sizes()
    weight_size = weight_shape[1]

    output_shape = outputs.type().sizes()
    out_h = output_shape[2]
    out_w = output_shape[3]

    kernel_size, groups = infer_kernel_size_and_groups(weight_size, channels)
    # print(kernel_size, groups)
    if dilation_map is not None:
        groups = dilation_map.type().sizes()[1]

    assert weight_size % groups == 0, "Weight channels must be divisible by number of groups."
    kh_kw = weight_size // groups

    # Total number of output elements
    num_output_elements = batch_size * channels * out_h * out_w

    # Operations per output element:
    if dilation_map is None:
        # Standard convolution (without dilation map):
        # For each output element, we perform k_h * k_w multiply-adds (2 FLOPs per position)
        flops_per_element = kh_kw * 2
    else:
        # Dynamic convolution with dilation map:
        # 1. Bilinear interpolation: ~8 FLOPs per sampling point
        #    (4 reads and 4 muls for corners, 3 adds for combining)
        # 2. Multiply-add with weight: 2 FLOPs
        flops_per_element = kh_kw * 10

    # Total FLOPs
    total_flops = num_output_elements * flops_per_element
    return total_flops


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")  # type: ignore
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")  # type: ignore
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")  # type: ignore
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")  # type: ignore
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")  # type: ignore
    flops += L * in_for_flops
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """

    return flops


def selective_scan_flop_jit(inputs, outputs):
    """
    JIT handle for SelectiveScanFn.
    """
    # xs, dts, As, Bs, Cs, Ds (skip), z (skip), dt_projs_bias (skip)
    # assert inputs[0].debugName().startswith("xs")  # (B, D, L)
    # assert inputs[2].debugName().startswith("As")  # (D, N)
    # assert inputs[3].debugName().startswith("Bs")  # (D, N)
    with_Group = len(inputs[3].type().sizes()) == 4
    with_D = inputs[5].debugName().startswith("Ds")
    if not with_D:
        with_z = inputs[5].debugName().startswith("z")
    else:
        with_z = inputs[6].debugName().startswith("z")
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_ref(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    return flops


def mamba_inner_fn_no_out_proj_flop_jit(inputs, outputs):
    """
    JIT handle for MambaInnerFnNoOutProj.
    """
    xz = inputs[0]
    conv1d_weight = inputs[1]
    x_proj_weight = inputs[3]
    delta_proj_weight = inputs[4]
    A = inputs[5]

    xz_shape = xz.type().sizes()  # (B, 2*D, L)
    B = xz_shape[0]
    L = xz_shape[-1]
    D = xz_shape[1] // 2

    K = conv1d_weight.type().sizes()[-1]  # d_conv
    N = A.type().sizes()[1]  # d_state
    rank = delta_proj_weight.type().sizes()[1]  # dt_rank

    flops = 0

    # 1. 1D Convolution (Depthwise: 2 * B * L * D * K)
    # 2. x_proj: (B, L, D) -> (B, L, rank + 2*N)
    # 3. dt_proj: (B, L, rank) -> (B, L, D)
    flops += 2 * B * L * D * K
    flops += 2 * (B * L) * D * (rank + 2 * N)
    flops += 2 * (B * L) * rank * D
    flops += flops_selective_scan_ref(B=B, L=L, D=D, N=N, with_D=True, with_Z=True, with_Group=True)
    return flops
