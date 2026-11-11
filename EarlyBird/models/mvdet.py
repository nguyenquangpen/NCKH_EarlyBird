import torch
import torch.nn as nn

import utils.geom
import utils.vox
import utils.basic

from kornia.filters import gaussian_blur2d
from kornia.geometry.transform.imgwarp import warp_perspective
from models.encoder import Encoder_res101, Encoder_res50, Encoder_res18, Encoder_eff, Encoder_swin_t, Encoder_res34
from models.decoder import Decoder

# view_fusion mode -> (gate, pool, multi_height)
#   gate:  None | 'geo' | 'm1'   per-cell camera weights w_s and the gate 1 + alpha (N w_s - 1) on the BEV features
#   pool:  None | 'geo' | 'w'    BEV Re-ID embedding += sum_s w_s * (image Re-ID embedding of camera s projected to the
#                                 BEV cell at body height fusion_heights[1]); w from the gate module (geo or learnt)
#   multi: bool                  project the camera features at every height in fusion_heights (not only the ground)
MODE_SPEC = {
    'none': (None, None, False),   # EarlyBird baseline
    'geo': ('geo', None, False),   # geometric weights only, alpha = 1
    'm1': ('m1', None, False),     # learnt view-aware gate
    'p0': (None, 'geo', False),    # baseline + view-pooled Re-ID with geometric weights (ablation: pooling alone)
    'm3': ('m1', 'w', False),      # M1 gate + view-pooled Re-ID with the M1 weights
    'm4': ('m1', 'w', True),       # m3 + multi-height feature projection
}
VIEW_FUSION_MODES = tuple(MODE_SPEC)


class ViewGatedFusion(nn.Module):
    """
    View-aware gated fusion of per-camera BEV features.

    geo: w_s = W_geo_s / sum_s' W_geo_s'                       (no parameters, alpha = 1)
    m1:  w_s = softmax_s(log W_geo_s + g([F_s, ray_s]) / T)    (g shared over cameras, alpha learnt from 0)
    Both: gate_s = 1 + alpha * (N * w_s - 1),  F'_s = gate_s * F_s
    Weights are normalised over the camera dimension and are exactly 0 for cameras that do not see a cell.
    """

    def __init__(self, feat_dim, mode='m1', hidden_dim=64, blur_kernel=5, blur_sigma=1.5, temperature=1.0):
        super().__init__()
        assert mode in ('geo', 'm1')
        self.mode = mode
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.temperature = temperature

        if mode == 'm1':
            self.g = nn.Sequential(
                nn.Conv2d(feat_dim + 2, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, 1, kernel_size=1),
            )
            # start from the geometric prior: g == 0 at init
            nn.init.zeros_(self.g[-1].weight)
            nn.init.zeros_(self.g[-1].bias)
            self.alpha = nn.Parameter(torch.zeros(1))
        else:
            self.g = None
            self.register_buffer('alpha', torch.ones(1), persistent=False)

    def forward(self, feats, vis, ray):
        """
        feats: (B,S,C,Y,X) per-camera BEV features
        vis:   (B,S,1,Y,X) bool, camera s sees cell (y,x)
        ray:   (B,S,2,Y,X) unit vector camera -> cell in mem coordinates
        returns gated feats (B,S,C,Y,X), N/S (B,1,Y,X), weights w (B,S,1,Y,X), gate (B,S,1,Y,X)
        """
        B, S, C, Y, X = feats.shape
        vis_f = vis.to(feats.dtype)
        w_geo = gaussian_blur2d(vis_f.reshape(B * S, 1, Y, X), (self.blur_kernel, self.blur_kernel),
                                (self.blur_sigma, self.blur_sigma)).reshape(B, S, 1, Y, X)
        count = vis_f.sum(dim=1, keepdim=True)  # B,1,1,Y,X

        if self.mode == 'geo':
            w_geo = w_geo * vis_f
            w = w_geo / w_geo.sum(dim=1, keepdim=True).clamp_min(1e-6)
        else:
            g_in = torch.cat([feats, ray.to(feats.dtype)], dim=2).reshape(B * S, C + 2, Y, X)
            logit = torch.log(w_geo.clamp_min(1e-6)) + self.g(g_in).reshape(B, S, 1, Y, X)
            logit = (logit / self.temperature).masked_fill(~vis, -1e4)
            # softmax over cameras (dim=1); cells seen by nobody give a uniform softmax, zeroed by vis
            w = torch.softmax(logit, dim=1) * vis_f

        gate = 1 + self.alpha * (count * w - 1)
        return gate * feats, count[:, 0] / S, w, gate


def pool_views(w, feats):
    """Convex combination over cameras: w (B,S,1,Y,X) sums to 1 over the cameras seeing a cell (0 elsewhere),
    feats (B,S,C,Y,X) -> (B,C,Y,X). Cells seen by no camera give 0."""
    return (w * feats).sum(dim=1)


class MVDet(nn.Module):
    def __init__(self, Y, Z, X,
                 rand_flip=False,
                 num_cameras=None,
                 num_ids=None,
                 latent_dim=512,
                 encoder_type='res18',
                 view_fusion='none',
                 fusion_heights=None,
                 device=None):
        """
        fusion_heights: heights above the ground plane in the ref frame's z unit (cm on Wildtrack, m on MultiviewX).
            [0] is the ground plane used for the BEV features; [1] is the body height at which the image Re-ID
            embeddings are sampled for the view-pooled Re-ID ('p0', 'm3', 'm4'); 'm4' projects the features at all.
        """
        super().__init__()
        assert (encoder_type in ['res101', 'res50', 'res18', 'res34', 'effb0', 'effb4', 'swin_t'])
        assert view_fusion in VIEW_FUSION_MODES, f'view_fusion must be one of {VIEW_FUSION_MODES}'
        assert view_fusion == 'none' or num_cameras is not None, 'view_fusion requires num_cameras'
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.Y, self.Z, self.X = Y, Z, X
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        self.num_cameras = num_cameras
        self.view_fusion = view_fusion
        self.gate_mode, self.pool_mode, self.multi_height = MODE_SPEC[view_fusion]
        self.heights = [float(h) for h in (fusion_heights if fusion_heights is not None else [0.0])]
        assert self.heights[0] == 0.0, 'fusion_heights[0] must be the ground plane (0)'
        if self.pool_mode is not None:
            assert len(self.heights) >= 2, f'view_fusion={view_fusion} needs fusion_heights=[0, body_height]'
            assert not rand_flip, 'view-pooled Re-ID is not implemented together with rand_flip'

        self.mean = torch.as_tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
        self.std = torch.as_tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)

        # Encoder
        self.feat2d_dim = 128
        if encoder_type == 'res101':
            self.encoder = Encoder_res101(self.feat2d_dim)
        elif encoder_type == 'res50':
            self.encoder = Encoder_res50(self.feat2d_dim)
        elif encoder_type == 'effb0':
            self.encoder = Encoder_eff(self.feat2d_dim, version='b0')
        elif encoder_type == 'res18':
            self.encoder = Encoder_res18(self.feat2d_dim)
        elif encoder_type == 'res34':
            self.encoder = Encoder_res34(self.feat2d_dim)
        elif encoder_type == 'swin_t':
            self.encoder = Encoder_swin_t(self.feat2d_dim)
        else:
            self.encoder = Encoder_eff(self.feat2d_dim, version='b4')

        if self.num_cameras is None:
            self.world_conv = nn.Sequential(
                nn.Conv3d(self.feat2d_dim, self.feat2d_dim, 3, padding=1),
                nn.InstanceNorm3d(latent_dim), nn.ReLU(),
                # nn.Conv3d(latent_dim, latent_dim, 3, padding=(1, 2, 2), dilation=(1, 2, 2)),
            )
        else:
            # gated modes append one N/S coverage channel; multi-height stacks one feature block per height
            extra_ch = 1 if self.gate_mode is not None else 0
            n_planes = len(self.heights) if self.multi_height else 1
            if self.gate_mode is not None or self.pool_mode is not None:
                self.fusion = ViewGatedFusion(self.feat2d_dim, mode=self.gate_mode or 'geo')
            self.world_feat = nn.Sequential(
                nn.Conv2d(self.feat2d_dim * self.num_cameras * n_planes + extra_ch, latent_dim, kernel_size=3,
                          padding=1),
                nn.InstanceNorm3d(latent_dim), nn.ReLU(),
                nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
            )

        # self.cam_out = nn.Sequential(
        #     nn.Conv2d(latent_dim, self.feat2d_dim, 3, padding=1),
        #     nn.InstanceNorm2d(self.feat2d_dim), nn.ReLU(),
        #     nn.Conv2d(self.feat2d_dim, self.feat2d_dim, 1, padding=0),
        # )

        self.decoder = Decoder(
            in_channels=latent_dim,
            n_classes=2,
            n_ids=num_ids,
        )

        # Weights
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.size_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.rot_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def view_geometry(self, proj_mats, cams_T_ref_, ref_T_cams_, ref_T_mem, vox_util, Hf, Wf):
        """
        Fixed (not learnt) per-camera geometry on the BEV grid.
        proj_mats: (B*S,3,3) mem_T_featpix, cams_T_ref_ / ref_T_cams_: (B*S,4,4), ref_T_mem: (3,3) ground plane
        returns vis_ (B*S,1,Y,X) bool, ray_ (B*S,2,Y,X) unit vector camera -> cell centre in mem (x, y)
        """
        BS, device, dtype = proj_mats.shape[0], proj_mats.device, proj_mats.dtype
        ones = torch.ones(BS, 1, Hf, Wf, device=device, dtype=dtype)
        in_image = warp_perspective(ones, proj_mats, (self.Y, self.X), align_corners=False) > 0.5

        ys, xs = torch.meshgrid(torch.arange(self.Y, device=device, dtype=dtype),
                                torch.arange(self.X, device=device, dtype=dtype), indexing='ij')
        grid = torch.stack([xs, ys], 0)  # 2,Y,X  (mem x, mem y) of each cell centre

        # the ground homography also maps cells behind the camera into the image; drop them
        z_coef = torch.matmul(cams_T_ref_[:, 2:3, [0, 1, 3]], ref_T_mem)[:, 0]  # B*S,3
        depth = z_coef[:, 0, None, None] * xs + z_coef[:, 1, None, None] * ys + z_coef[:, 2, None, None]
        # "in front" is the depth sign of most in-image cells: real cameras (Wildtrack) look along +z,
        # MultiviewX's game-engine calibration looks along -z (visible ground has negative camera depth)
        in_img = in_image[:, 0]
        n_pos = (in_img & (depth > 0)).sum((1, 2))
        n_neg = (in_img & (depth < 0)).sum((1, 2))
        front = torch.where(n_neg > n_pos, -torch.ones_like(depth[:, 0, 0]), torch.ones_like(depth[:, 0, 0]))
        vis_ = in_image & (depth * front[:, None, None] > 0).unsqueeze(1)

        mem_T_ref = vox_util.get_mem_T_ref(1, self.Y, self.Z, self.X, device=device)[0].to(dtype)  # 4,4
        cam_mem = utils.geom.apply_4x4(mem_T_ref.expand(BS, 4, 4), ref_T_cams_[:, None, :3, 3])[:, 0]  # B*S,3
        ray_ = grid.unsqueeze(0) - cam_mem[:, :2, None, None]  # ground projection of the camera centre
        ray_ = ray_ / ray_.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return vis_, ray_

    @staticmethod
    def plane_projection(featpix_T_cams_, cams_T_ref_, ref_T_mem, h=0.0):
        """
        mem_T_featpix (B*S,3,3): homography from feature pixels to BEV cells for the plane z = h of the ref frame.
        A ref point (x, y, h) maps to camera coordinates with columns 0, 1 of cams_T_ref plus h * column 2 + column 3,
        so h = 0 is the ground-plane homography of the baseline.
        """
        cols = cams_T_ref_[:, :3, [0, 1, 3]].clone()
        cols[:, :, 2] = cols[:, :, 2] + h * cams_T_ref_[:, :3, 2]
        featpix_T_ref_ = torch.matmul(featpix_T_cams_[:, :3, :3], cols)  # B*S,3,3
        return torch.inverse(torch.matmul(featpix_T_ref_, ref_T_mem))

    def forward(self, rgb_cams, pix_T_cams, cams_T_global, vox_util, ref_T_global):
        """
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_cams: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cams_T_global: (B,S,4,4)
        vox_util: vox util object
        ref_T_global: (B,4,4)
        """
        B, S, C, H, W = rgb_cams.shape
        # reshape tensors
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        rgb_cams_ = __p(rgb_cams)  # B*S,3,H,W
        pix_T_cams_ = __p(pix_T_cams)  # B*S,4,4
        cams_T_global_ = __p(cams_T_global)  # B*S,4,4

        global_T_cams_ = torch.inverse(cams_T_global_)  # B*S,4,4
        ref_T_cams_ = torch.matmul(ref_T_global.repeat(S, 1, 1), global_T_cams_)  # B*S,4,4
        cams_T_ref_ = torch.inverse(ref_T_cams_)  # B*S,4,4

        # rgb encoder
        device = rgb_cams_.device
        rgb_cams_ = (rgb_cams_ - self.mean.to(device)) / self.std.to(device)  # B*S,3,H,W
        feat_cams_ = self.encoder(rgb_cams_)  # B*S,latent_dim,H/8,W/8
        _, C, Hf, Wf = feat_cams_.shape
        sy = Hf / float(H)
        sx = Wf / float(W)
        featpix_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)  # B*S,4,4

        ref_T_mem = vox_util.get_ref_T_mem(B, self.Y, self.Z, self.X, device=device)  # B,4,4
        ref_T_mem = ref_T_mem[0, [0, 1, 3]][:, [0, 1, 3]]  # 3,3
        # feature pixels -> BEV cells for every plane z = h of the ref frame; index 0 is the ground plane
        proj_mats_h = [self.plane_projection(featpix_T_cams_, cams_T_ref_, ref_T_mem, h) for h in self.heights]
        proj_mats = proj_mats_h[0]  # B*S,3,3

        planes = proj_mats_h if self.multi_height else proj_mats_h[:1]
        # one B*S,latent_dim,Y,X block per projected plane
        world_features_ = [warp_perspective(feat_cams_, P, (self.Y, self.X), align_corners=False) for P in planes]
        world_features = __u(world_features_[0])  # B,S,latent_dim,Y,X (ground plane)

        id_pool, img_id_feat_ = None, None
        if self.num_cameras is None:
            world_features = self.world_conv(world_features.permute(0, 2, 1, 3, 4))
            world_features = self.world_feat(world_features.sum(2, keepdim=True))
        elif self.view_fusion == 'none':
            world_features = self.world_feat(world_features.view(B, S * self.feat2d_dim, self.Y, self.X))
        else:
            with torch.no_grad():
                vis_, ray_ = self.view_geometry(proj_mats, cams_T_ref_, ref_T_cams_, ref_T_mem, vox_util, Hf, Wf)
            gated, coverage, w, gate = self.fusion(world_features, __u(vis_), __u(ray_))
            if self.pool_mode is not None:
                # image Re-ID embeddings sampled where a person standing on the cell has its body (heights[1]),
                # pooled over the cameras with the view weights; the Re-ID losses train w through this path
                img_id_feat_ = self.decoder.img_id_embed(feat_cams_)  # B*S,reid_feat,Hf,Wf
                id_proj = warp_perspective(img_id_feat_, proj_mats_h[1], (self.Y, self.X), align_corners=False)
                id_pool = pool_views(w, __u(id_proj))  # B,reid_feat,Y,X
            if self.gate_mode is None:  # 'p0': the baseline aggregation, the weights only pool the Re-ID
                world_features = self.world_feat(world_features.reshape(B, S * self.feat2d_dim, self.Y, self.X))
            else:
                blocks = [gated] + [gate * __u(f) for f in world_features_[1:]]  # the same gate on every plane
                world_features = torch.cat([b.reshape(B, S * self.feat2d_dim, self.Y, self.X) for b in blocks]
                                           + [coverage], 1)
                world_features = self.world_feat(world_features)

        # back_proj_mats = torch.inverse(proj_mats)  # B*S,3,3
        # world_features_cam = __p(world_features.unsqueeze(1).repeat(1, S, 1, 1, 1))
        # feat_cams_back_ = warp_perspective(world_features_cam, back_proj_mats, (Hf, Wf), align_corners=False)
        # feat_cams_ = feat_cams_ + self.cam_out(feat_cams_back_)

        out_dict = self.decoder(world_features, feat_cams_,
                                (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None,
                                id_pool=id_pool, img_id_feat=img_id_feat_)

        return out_dict
