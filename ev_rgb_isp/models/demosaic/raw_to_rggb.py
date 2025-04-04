import torch


def quad_raw_to_rggb(raw: torch.Tensor):
    b, c, h, w = raw.size()
    assert h % 4 == 0 and w % 4 == 0
    r = torch.zeros((b, c, h // 2, w // 2), device=raw.device)
    g1 = torch.zeros((b, c, h // 2, w // 2), device=raw.device)
    g2 = torch.zeros((b, c, h // 2, w // 2), device=raw.device)
    b = torch.zeros((b, c, h // 2, w // 2), device=raw.device)
    # R
    r[:, :, 0::2, 0::2] = raw[:, :, 0::4, 0::4]
    r[:, :, 0::2, 1::2] = raw[:, :, 0::4, 1::4]
    r[:, :, 1::2, 0::2] = raw[:, :, 1::4, 0::4]
    r[:, :, 1::2, 1::2] = raw[:, :, 1::4, 1::4]
    # G1
    g1[:, :, 0::2, 0::2] = raw[:, :, 0::4, 2::4]
    g1[:, :, 0::2, 1::2] = raw[:, :, 0::4, 3::4]
    g1[:, :, 1::2, 0::2] = raw[:, :, 1::4, 2::4]
    g1[:, :, 1::2, 1::2] = raw[:, :, 1::4, 3::4]
    # G2
    g2[:, :, 0::2, 0::2] = raw[:, :, 2::4, 0::4]
    g2[:, :, 0::2, 1::2] = raw[:, :, 2::4, 1::4]
    g2[:, :, 1::2, 0::2] = raw[:, :, 3::4, 0::4]
    g2[:, :, 1::2, 1::2] = raw[:, :, 3::4, 1::4]
    # B
    b[:, :, 0::2, 0::2] = raw[:, :, 2::4, 2::4]
    b[:, :, 0::2, 1::2] = raw[:, :, 2::4, 3::4]
    b[:, :, 1::2, 0::2] = raw[:, :, 3::4, 2::4]
    b[:, :, 1::2, 1::2] = raw[:, :, 3::4, 3::4]
    # B, 1, H, W -> B, 4, 1, H, W
    rggb = torch.stack([r, g1, g2, b], dim=1)
    return rggb
