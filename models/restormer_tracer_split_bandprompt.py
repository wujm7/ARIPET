#网络结构代码

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------
# 1. 改进的 CoordConcat (保持你的逻辑，增加适配性)
# ---------------------------------------------------------------
class CoordConcat(nn.Module):
    def __init__(self, with_r=True):
        super(CoordConcat, self).__init__()
        self.with_r = with_r

    def forward(self, x, spatial_width):
        B, C, H, W = x.shape
        # Nyquist-normalized coordinates in the SAME unshifted rfft2 order.
        y = 2 * torch.fft.fftfreq(H, device=x.device, dtype=x.dtype)
        fx = 2 * torch.fft.rfftfreq(spatial_width, device=x.device, dtype=x.dtype)
        if len(fx) != W:
            raise ValueError("Half-spectrum shape does not match spatial width")
        y_coords = y.view(1, 1, H, 1).expand(B, 1, H, W)
        x_coords = fx.view(1, 1, 1, W).expand(B, 1, H, W)

        coords = [x, y_coords, x_coords]

        if self.with_r:
            r_coords = torch.sqrt(y_coords ** 2 + x_coords ** 2)
            coords.append(r_coords)

        return torch.cat(coords, dim=1)


# ---------------------------------------------------------------
# 2. 改进的 PALHBlock (核心修改：支持任意 dim，自动 FFT/IFFT)
# ---------------------------------------------------------------
class PALHBlock(nn.Module):
    def __init__(self, dim):
        super(PALHBlock, self).__init__()
        coord_dim = 3
        self.coord = CoordConcat(with_r=True)

        # low-frequency amplitude branch
        self.amp_low_conv = nn.Sequential(
            nn.Conv2d(dim + coord_dim, dim, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, 1)
        )

        # high-frequency amplitude branch
        self.amp_high_conv = nn.Sequential(
            nn.Conv2d(dim + coord_dim, dim, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, 1)
        )

        self.phase_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, 1)
        )

    def _build_band_masks(self, h, w, device, dtype):
        # h,w are spatial dimensions: retain w to handle odd widths correctly.
        yy = (2 * torch.fft.fftfreq(h, device=device, dtype=dtype)).view(h, 1)
        xx = (2 * torch.fft.rfftfreq(w, device=device, dtype=dtype)).view(1, -1)
        rr = torch.sqrt(yy ** 2 + xx ** 2)
        low_mask = (rr <= 0.5).to(dtype).view(1, 1, h, w // 2 + 1)
        high_mask = 1.0 - low_mask
        return low_mask, high_mask

    def forward(self, x, prompt=None):
        fft_x = torch.fft.rfft2(x, norm='ortho')
        amp = torch.abs(fft_x)
        phase = torch.angle(fft_x)

        amp_mean = amp.mean(dim=(2, 3), keepdim=True)
        amp_std = amp.std(dim=(2, 3), keepdim=True, unbiased=False)
        amp_norm = (amp - amp_mean) / (amp_std + 1e-9)

        amp_with_coord = self.coord(amp_norm, spatial_width=x.shape[-1])
        amp_low = self.amp_low_conv(amp_with_coord)
        amp_high = self.amp_high_conv(amp_with_coord)

        low_mask, high_mask = self._build_band_masks(x.shape[-2], x.shape[-1], amp.device, amp.dtype)

        if prompt is not None:
            prompt_freq = torch.fft.rfft2(prompt, s=x.shape[-2:], norm='ortho')
            prompt_amp = torch.abs(prompt_freq)
            prompt_gate = torch.sigmoid(prompt_amp)
            low_gate = prompt_gate * low_mask
            high_gate = prompt_gate * high_mask
        else:
            low_gate = low_mask
            high_gate = high_mask

        amp_processed = amp_low * low_gate + amp_high * high_gate
        amp_out = amp + amp_processed * amp_std

        phase_processed = self.phase_conv(phase)
        phase_out = phase + phase_processed

        complex_out = torch.complex(amp_out * torch.cos(phase_out), amp_out * torch.sin(phase_out))
        out = torch.fft.irfft2(complex_out, s=x.shape[-2:], norm='ortho')
        return out

##########################################################################
## Layer Norm (保持不变)
# ... (BiasFree_LayerNorm, WithBias_LayerNorm, LayerNorm 代码省略，保持原样) ...
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN) (保持不变)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA) (保持不变)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


### ----------------------------------------------------------------------
### PROMPT MODIFICATION: 添加 PromptGenBlock
### ----------------------------------------------------------------------
class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=64, prompt_len=5, prompt_size=32, lin_dim=96):
        """
        prompt_dim: Prompt 的通道数 (应与当前层特征通道数一致)
        prompt_len: Prompt 的组件数量 (如 5 或 10)
        prompt_size: 学习到的 Prompt 的空间尺寸 (会插值)
        lin_dim: 输入特征图的通道数 (用于计算权重，通常等于 prompt_dim)
        """
        super(PromptGenBlock, self).__init__()
        self.prompt_len = prompt_len
        # Parameter: (1, Components, Channels, H, W)
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        # 1. 计算权重: Global Average Pooling -> Linear -> Softmax
        emb = x.mean(dim=(-2, -1))  # (B, C)
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)  # (B, prompt_len)

        # 2. 组合 Prompt: Weights * Prompt_Params
        # (B, prompt_len, 1, 1, 1) * (1, prompt_len, dim, h, w) -> (B, prompt_len, dim, h, w)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param

        # Sum over components dimension -> (B, dim, h, w)
        prompt = torch.sum(prompt, dim=1)

        # 3. 调整尺寸并精炼
        # 将 Prompt 插值到当前特征图的大小 (H, W)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear", align_corners=False)
        prompt = self.conv3x3(prompt)

        return prompt


# ... (保留之前的 TransformerBlock, PromptGenBlock 等定义) ...

class Restormer(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False,
                 use_prompt=True,
                 prompt_len=10,
                 num_tracers=13,
                 spatial_prompt=True,
                 fourier_prompt=True
                 ):
        super(Restormer, self).__init__()
        self.use_prompt = use_prompt
        self.spatial_prompt = spatial_prompt
        self.fourier_prompt = fourier_prompt

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # Encoder Layers (保持不变)
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim * 2 ** 2))

        # -----------------------------------------------------
        # Latent Layer (Level 4) - 融合 PALHBlock
        # -----------------------------------------------------
        latent_dim = int(dim * 2 ** 3)
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=latent_dim, num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        # PALH 频域增强模块
        self.palh_l4 = PALHBlock(dim=latent_dim)

        # if self.use_prompt:
        #     self.prompt_l4 = PromptGenBlock(prompt_dim=latent_dim, prompt_len=prompt_len, lin_dim=latent_dim)
        #     # 融合层: 将 [Feature, Prompt-Gated-Freq] 融合
        #     self.fusion_l4 = nn.Conv2d(latent_dim * 2, latent_dim, 1, bias=bias)
        if self.use_prompt:
            self.prompt_l4 = PromptGenBlock(prompt_dim=latent_dim, prompt_len=prompt_len, lin_dim=latent_dim)
            # 空间域 prompt 注入
            self.spatial_fusion_l4 = nn.Conv2d(latent_dim * 2, latent_dim, 1, bias=bias)
            # 最终融合 [spatial_feat, freq_feat]
            self.fusion_l4 = nn.Conv2d(latent_dim * 2, latent_dim, 1, bias=bias)
        # shared / specific split at bottleneck
        self.shared_head = nn.Conv2d(latent_dim, latent_dim, kernel_size=1, bias=bias)
        self.specific_head = nn.Conv2d(latent_dim, latent_dim, kernel_size=1, bias=bias)
        self.shared_specific_fusion = nn.Conv2d(latent_dim * 2, latent_dim, kernel_size=1, bias=bias)
        self.tracer_pool = nn.AdaptiveAvgPool2d(1)
        self.tracer_classifier = nn.Linear(latent_dim, num_tracers)
        # -----------------------------------------------------

        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        # -----------------------------------------------------
        # Decoder Level 3 - 融合 PALHBlock
        # -----------------------------------------------------
        l3_dim = int(dim * 2 ** 2)
        self.palh_l3 = PALHBlock(dim=l3_dim)

        # if self.use_prompt:
        #     self.prompt_l3 = PromptGenBlock(prompt_dim=l3_dim, prompt_len=prompt_len, lin_dim=l3_dim)
        #     self.fusion_l3 = nn.Conv2d(l3_dim * 2, l3_dim, 1, bias=bias)
        if self.use_prompt:
            self.prompt_l3 = PromptGenBlock(prompt_dim=l3_dim, prompt_len=prompt_len, lin_dim=l3_dim)

            # 空间域 prompt 注入
            self.spatial_fusion_l3 = nn.Conv2d(l3_dim * 2, l3_dim, 1, bias=bias)

            # 最终融合 [spatial_feat_3, freq_feat_3]
            self.fusion_l3 = nn.Conv2d(l3_dim * 2, l3_dim, 1, bias=bias)
        # -----------------------------------------------------

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        # -----------------------------------------------------
        # Decoder Level 2 - 融合 PALHBlock
        # -----------------------------------------------------
        l2_dim = int(dim * 2 ** 1)
        self.palh_l2 = PALHBlock(dim=l2_dim)


        if self.use_prompt:
            self.prompt_l2 = PromptGenBlock(prompt_dim=l2_dim, prompt_len=prompt_len, lin_dim=l2_dim)

            # 空间域 prompt 注入
            self.spatial_fusion_l2 = nn.Conv2d(l2_dim * 2, l2_dim, 1, bias=bias)

            # 最终融合 [spatial_feat_2, freq_feat_2]
            self.fusion_l2 = nn.Conv2d(l2_dim * 2, l2_dim, 1, bias=bias)
        # -----------------------------------------------------

        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        # Encoder (Standard)
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        # shared / specific split
        z_sh = self.shared_head(latent)
        z_sp = self.specific_head(latent)
        cls_feat = self.tracer_pool(z_sp).flatten(1)
        tracer_logits = self.tracer_classifier(cls_feat)
        latent_core = self.shared_specific_fusion(torch.cat([z_sh, z_sp], dim=1))

        # --- Level 4: Spatial Prompt Injection + Frequency Prompt Modulation ---
        if self.use_prompt:
            prompt = self.prompt_l4(z_sp)
            # 1) 空间域 prompt 注入
            spatial_feat = self.spatial_fusion_l4(torch.cat([latent_core, prompt], dim=1)) if self.spatial_prompt else latent_core
            # 2) 频域 prompt 调制
            freq_feat = self.palh_l4(latent_core, prompt=prompt if self.fourier_prompt else None)
            latent = torch.cat([spatial_feat, freq_feat], dim=1)
            latent = self.fusion_l4(latent)
        else:
            freq_feat = self.palh_l4(latent_core, prompt=None)
            latent = latent_core + freq_feat
        # ----------------------------------------------------------------------
        # ----------------------------------------------

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        # --- Level 3: Prompt-Gated Frequency Fusion ---
        if self.use_prompt:
            prompt_3 = self.prompt_l3(out_dec_level3)
            # 1) 空间域 prompt 注入
            spatial_feat_3 = self.spatial_fusion_l3(torch.cat([out_dec_level3, prompt_3], dim=1)) if self.spatial_prompt else out_dec_level3
            # 2) 频域 prompt 调制
            freq_feat_3 = self.palh_l3(out_dec_level3, prompt=prompt_3 if self.fourier_prompt else None)
            # 3) 空间域和频域融合
            out_dec_level3 = torch.cat([spatial_feat_3, freq_feat_3], dim=1)
            out_dec_level3 = self.fusion_l3(out_dec_level3)
        else:
            freq_feat_3 = self.palh_l3(out_dec_level3, prompt=None)
            out_dec_level3 = out_dec_level3 + freq_feat_3
        # ----------------------------------------------

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        # --- Level 2: Prompt-Gated Frequency Fusion ---
        if self.use_prompt:
            prompt_2 = self.prompt_l2(out_dec_level2)
            # 1) 空间域 prompt 注入
            spatial_feat_2 = self.spatial_fusion_l2(torch.cat([out_dec_level2, prompt_2], dim=1)) if self.spatial_prompt else out_dec_level2
            # 2) 频域 prompt 调制
            freq_feat_2 = self.palh_l2(out_dec_level2, prompt=prompt_2 if self.fourier_prompt else None)
            # 3) 空间域和频域融合
            out_dec_level2 = torch.cat([spatial_feat_2, freq_feat_2], dim=1)
            out_dec_level2 = self.fusion_l2(out_dec_level2)
        else:
            freq_feat_2 = self.palh_l2(out_dec_level2, prompt=None)
            out_dec_level2 = out_dec_level2 + freq_feat_2
        # ----------------------------------------------

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1, tracer_logits, z_sh, z_sp



from models.pet_objective import compute_orth_loss, compute_total_loss

if __name__ == '__main__':
    # ---------------------------------------------------------------
    # 测试 Prompt-Restormer
    H, W = 128, 128
    channels = 1
    batch_size = 1

    print(f"正在创建 Prompt-Restormer 模型...")

    # 实例化模型
    model = Restormer(
        inp_channels=channels,
        out_channels=channels,
        dim=48,
        use_prompt=True,  # 开启 Prompt
        prompt_len=10
    )

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params / 1e6:.2f} M")

    # 创建随机输入张量
    input_tensor = torch.randn(batch_size, channels, H, W)

    # 检查 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # 前向传播
    model.eval()
    with torch.no_grad():
        output_tensor, tracer_logits, z_sh, z_sp = model(input_tensor)

    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output_tensor.shape}")
    print(f"tracer logits 形状: {tracer_logits.shape}")
    print(f"z_sh 形状: {z_sh.shape}")
    print(f"z_sp 形状: {z_sp.shape}")

    if input_tensor.shape == output_tensor.shape:
        print("✅ Prompt 集成测试通过。")
    else:
        print("❌ 尺寸不匹配！")
