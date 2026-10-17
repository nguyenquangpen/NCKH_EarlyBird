"""
CPU tests for the view-aware gated fusion (models/mvdet.py, view_fusion = none | geo | m1).
Run from the EarlyBird/ package dir:  python -m pytest tests/test_view_fusion.py -q
"""
import importlib.util
import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.mvdet import MVDet, ViewGatedFusion  # noqa: E402
from utils import vox  # noqa: E402

B, S, C, Y, X = 2, 4, 8, 20, 30
DEVICE = torch.device('cpu')


class CpuVoxelUtil(vox.VoxelUtil):
    # the original MVDet calls get_ref_T_mem without a device (defaults to cuda)
    def get_ref_T_mem(self, B, Y, Z, X, assert_cube=False, device='cpu'):
        return super().get_ref_T_mem(B, Y, Z, X, assert_cube=assert_cube, device='cpu')


def random_fusion_inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(B, S, C, Y, X, generator=g)
    vis = torch.rand(B, S, 1, Y, X, generator=g) > 0.4
    vis[:, :, :, :3, :3] = False  # a block seen by nobody
    vis[:, :, :, -3:, -3:] = True  # a block seen by everybody
    ray = torch.randn(B, S, 2, Y, X, generator=g)
    ray = ray / ray.norm(dim=2, keepdim=True)
    return feats, vis, ray


def randomise_g(fusion, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in fusion.g.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)


# ---------------------------------------------------------------- (a) normalisation over cameras
@pytest.mark.parametrize('mode', ['geo', 'm1'])
def test_weights_sum_to_one_over_visible_cameras(mode):
    feats, vis, ray = random_fusion_inputs()
    fusion = ViewGatedFusion(C, mode=mode)
    if mode == 'm1':
        randomise_g(fusion)
    _, coverage, w = fusion(feats, vis, ray)

    count = vis.sum(dim=1)  # B,1,Y,X
    total = w.sum(dim=1)
    assert torch.allclose(total[count > 0], torch.ones_like(total[count > 0]), atol=1e-5)
    assert torch.all(total[count == 0] == 0)
    assert torch.all(w[~vis] == 0)
    assert torch.all(w >= 0)
    assert torch.allclose(coverage, count.float() / S)

    # soft weights, not a binary mask (the SVCW pitfall of normalising over a size-1 channel dim)
    shared = (count >= 2).unsqueeze(1).expand_as(w) & vis
    assert ((w[shared] > 0) & (w[shared] < 1)).float().mean() > 0.9


def test_m1_g_changes_weights():
    feats, vis, ray = random_fusion_inputs()
    fusion = ViewGatedFusion(C, mode='m1')
    _, _, w0 = fusion(feats, vis, ray)
    randomise_g(fusion)
    _, _, w1 = fusion(feats, vis, ray)
    assert not torch.allclose(w0, w1)


def test_geo_gate_is_n_times_w():
    feats, vis, ray = random_fusion_inputs()
    fusion = ViewGatedFusion(C, mode='geo')
    assert len(list(fusion.parameters())) == 0
    fused, coverage, w = fusion(feats, vis, ray)
    assert torch.allclose(fused, coverage.unsqueeze(1) * S * w * feats, atol=1e-6)


# ---------------------------------------------------------------- (b) alpha = 0 is the identity
def test_m1_alpha_zero_is_identity():
    feats, vis, ray = random_fusion_inputs()
    fusion = ViewGatedFusion(C, mode='m1')
    randomise_g(fusion)
    assert fusion.alpha.item() == 0
    fused, _, _ = fusion(feats, vis, ray)
    assert torch.equal(fused, feats)


# ---------------------------------------------------------------- full model on a synthetic rig
MY, MZ, MX = 16, 2, 48
BOUNDS = (0, 48, 0, 16, 0, 2)
H, W = 64, 128
NUM_IDS = 10


def look_at(eye, target):
    eye, target = torch.tensor(eye, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    fwd = target - eye
    fwd = fwd / fwd.norm()
    right = torch.linalg.cross(fwd, torch.tensor([0., 0., 1.]))
    right = right / right.norm()
    down = torch.linalg.cross(fwd, right)
    R = torch.stack([right, down, fwd])  # rows: cam axes in global
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ eye
    return T  # cam_T_global


def synthetic_batch(batch=1, num_cams=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    eyes = [(-5., -5., 10.), (53., -5., 10.), (24., 25., 10.), (24., -8., 6.)][:num_cams]
    extr = torch.stack([look_at(e, (24., 8., 0.)) for e in eyes])
    intr = torch.eye(4)
    intr[0, 0] = intr[1, 1] = W / 2
    intr[0, 2], intr[1, 2] = W / 2, H / 2
    return dict(
        rgb_cams=torch.rand(batch, num_cams, 3, H, W, generator=g),
        pix_T_cams=intr.expand(batch, num_cams, 4, 4).clone(),
        cams_T_global=extr.expand(batch, num_cams, 4, 4).clone(),
        ref_T_global=torch.eye(4).expand(batch, 4, 4).clone(),
    )


def make_vox_util():
    return CpuVoxelUtil(MY, MZ, MX, scene_centroid=torch.zeros(1, 3), bounds=BOUNDS)


def make_model(mode, num_cams=3):
    torch.manual_seed(0)
    return MVDet(MY, MZ, MX, num_cameras=num_cams, num_ids=NUM_IDS, view_fusion=mode, device=DEVICE).eval()


# ---------------------------------------------------------------- (c) shapes
@pytest.mark.parametrize('mode', ['none', 'geo', 'm1'])
def test_full_model_shapes(mode):
    num_cams, batch = 3, 1
    model = make_model(mode, num_cams)
    expected_in = 128 * num_cams + (0 if mode == 'none' else 1)
    assert model.world_feat[0].in_channels == expected_in

    seen = {}
    model.world_feat.register_forward_hook(lambda m, i, o: seen.update(inp=i[0], out=o))
    with torch.no_grad():
        out = model(vox_util=make_vox_util(), **synthetic_batch(batch, num_cams))

    assert seen['inp'].shape == (batch, expected_in, MY, MX)
    assert seen['out'].shape == (batch, model.latent_dim, MY, MX)
    assert out['instance_center'].shape == (batch, 1, MY, MX)
    assert out['instance_offset'].shape == (batch, 2, MY, MX)
    assert out['img_center'].shape[0] == batch * num_cams
    for v in out.values():
        assert torch.isfinite(v).all()


def test_view_geometry_on_rig():
    num_cams = 3
    model = make_model('geo', num_cams)
    vis = {}
    model.fusion.register_forward_hook(lambda m, i, o: vis.update(vis=i[1], ray=i[2], w=o[2]))
    with torch.no_grad():
        model(vox_util=make_vox_util(), **synthetic_batch(1, num_cams))
    assert vis['vis'].shape == (1, num_cams, 1, MY, MX) and vis['vis'].dtype == torch.bool
    assert vis['vis'].any(dim=1).float().mean() > 0.5  # the rig covers most of the ground
    assert torch.allclose(vis['ray'].norm(dim=2), torch.ones(1, num_cams, MY, MX), atol=1e-5)
    # camera 0 sits at global (-5,-5): rays to the far corner point towards +x,+y
    assert (vis['ray'][0, 0, :, -1, -1] > 0).all()


def test_m1_alpha_zero_full_model_world_features_unchanged():
    num_cams = 3
    model = make_model('m1', num_cams)
    randomise_g(model.fusion)
    io = {}
    model.fusion.register_forward_hook(lambda m, i, o: io.update(feats=i[0], fused=o[0]))
    model.world_feat.register_forward_hook(lambda m, i, o: io.update(wf_in=i[0]))
    with torch.no_grad():
        model(vox_util=make_vox_util(), **synthetic_batch(1, num_cams))
    assert torch.equal(io['fused'], io['feats'])
    assert torch.equal(io['wf_in'][:, :128 * num_cams], io['feats'].reshape(1, 128 * num_cams, MY, MX))


# ---------------------------------------------------------------- none == original code
def load_original_mvdet():
    try:
        src = subprocess.run(['git', 'show', 'fee7379:EarlyBird/models/mvdet.py'], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('original mvdet.py not available from git')
    spec = importlib.util.spec_from_loader('mvdet_original', loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(src, 'mvdet_original.py', 'exec'), module.__dict__)
    return module.MVDet


def test_none_matches_original_code():
    OrigMVDet = load_original_mvdet()
    num_cams = 3
    new = make_model('none', num_cams)
    orig = OrigMVDet(MY, MZ, MX, num_cameras=num_cams, num_ids=NUM_IDS, device=DEVICE).eval()
    assert list(new.state_dict().keys()) == list(orig.state_dict().keys())
    orig.load_state_dict(new.state_dict())

    batch = synthetic_batch(1, num_cams)
    with torch.no_grad():
        out_new = new(vox_util=make_vox_util(), **batch)
        out_orig = orig(vox_util=make_vox_util(), **batch)
    for k in out_orig:
        assert torch.equal(out_new[k], out_orig[k]), k

