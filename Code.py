# tiny_swin_denoise_cifar10.py
# Tiny Swin denoiser on CIFAR-10. Reports PSNR instead of accuracy.

import math, time, os
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# -------------------------
# Helper functions (windowing etc.)
# -------------------------
def window_partition(x, win_h, win_w):
    B, H, W, C = x.shape
    x = x.view(B, H // win_h, win_h, W // win_w, win_w, C)
    x = x.permute(0,1,3,2,4,5).contiguous()
    windows = x.view(-1, win_h, win_w, C)
    return windows

def window_reverse(windows, win_h, win_w, H, W):
    B = int(windows.shape[0] // (H//win_h * W//win_w))
    x = windows.view(B, H//win_h, W//win_w, win_h, win_w, -1)
    x = x.permute(0,1,3,2,4,5).contiguous()
    x = x.view(B, H, W, -1)
    return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = 0. if drop_prob is None else drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=48, patch_size=2):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_size = patch_size
    def forward(self, x):
        x = self.proj(x)  # B, C, H', W'
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1,2)  # B, L, C
        x = self.norm(x)
        x = x.view(B, H, W, C)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size: Tuple[int,int], qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        Wh, Ww = window_size
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2*Wh-1)*(2*Ww-1), num_heads))
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1,2,0).contiguous()
        relative_coords[:, :, 0] += Wh - 1
        relative_coords[:, :, 1] += Ww - 1
        relative_coords[:, :, 0] *= 2*Ww - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
    def forward(self, x, mask: Optional[torch.Tensor]=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)
        q = q.permute(0,2,1,3); k = k.permute(0,2,1,3); v = v.permute(0,2,1,3)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn + relative_position_bias
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(-1, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = torch.softmax(attn, dim=-1)
        else:
            attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=4, shift_size=0, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads=num_heads, window_size=(window_size, window_size),
                                    qkv_bias=True, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim*mlp_ratio), drop=drop)
    def forward(self, x, attn_mask: Optional[torch.Tensor]):
        B, H, W, C = x.shape
        x_skip = x
        x = x.view(B, H*W, -1)
        x = self.norm1(x)
        x = x.view(B, H, W, -1)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size, self.window_size)
        x_windows = x_windows.view(-1, self.window_size*self.window_size, x.shape[-1])
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, x.shape[-1])
        shifted_x = window_reverse(attn_windows, self.window_size, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1,2))
        else:
            x = shifted_x
        x = x.view(B, H*W, -1)
        x = x_skip.view(B, H*W, -1) + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.view(B, H, W, -1)
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.reduction = nn.Linear(4*input_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(4*input_dim)
    def forward(self, x):
        B, H, W, C = x.shape
        assert H % 2 == 0 and W % 2 == 0, "H and W must be even"
        x0 = x[:, 0::2, 0::2, :]; x1 = x[:, 1::2, 0::2, :]; x2 = x[:, 0::2, 1::2, :]; x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # B, H/2, W/2, 4*C
        x = x.view(B, -1, 4*C)
        x = self.norm(x)
        x = self.reduction(x)
        H, W = H//2, W//2
        x = x.view(B, H, W, -1)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, shift, downsample=None, drop_path_rate=0.):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift_size = 0 if (i % 2 == 0) else shift
            self.blocks.append(SwinBlock(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size, drop_path=drop_path_rate))
        self.downsample = downsample
        self.window_size = window_size
        self.shift = shift
    def create_attn_mask(self, H, W, window_size, shift_size, device):
        if shift_size == 0:
            return None
        img_mask = torch.zeros((1, H, W, 1), device=device)
        cnt = 0
        for h in (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None)):
            for w in (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None)):
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, window_size, window_size)
        mask_windows = mask_windows.view(-1, window_size*window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask
    def forward(self, x):
        B, H, W, C = x.shape
        device = x.device
        for blk in self.blocks:
            attn_mask = None
            if blk.shift_size > 0:
                attn_mask = self.create_attn_mask(H, W, blk.window_size, blk.shift_size, device)
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

# -------------------------
# Tiny swin denoiser model (reconstruction head)
# -------------------------
class TinySwinDenoiser(nn.Module):
    def __init__(self,
                 in_chans=3,
                 embed_dim=48,
                 depths=[2,2],
                 num_heads=[2,4],
                 window_size=4,
                 patch_size=2,
                 mlp_ratio=4.0,
                 drop_rate=0.0,
                 drop_path_rate=0.1,
                 img_size=32):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim, patch_size=patch_size)
        self.P = patch_size
        self.img_size = img_size
        in_dim = embed_dim
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i_stage in range(len(depths)):
            depth = depths[i_stage]
            num_head = num_heads[i_stage]
            downsample = PatchMerging(in_dim, in_dim*2) if (i_stage < len(depths)-1) else None
            layer = BasicLayer(dim=in_dim, depth=depth, num_heads=num_head, window_size=window_size,
                               shift=window_size//2, downsample=downsample,
                               drop_path_rate=dp_rates[cur:cur+depth][-1] if depth>0 else 0.0)
            self.stages.append(layer)
            cur += depth
            if i_stage < len(depths)-1:
                in_dim = in_dim * 2
        self.norm = nn.LayerNorm(in_dim)
        
        # Calculate upsampling factor needed
        # After patch_embed: img_size/patch_size
        # After len(depths)-1 downsamples: img_size/(patch_size * 2^(len(depths)-1))
        num_downsamples = len(depths) - 1
        final_size = img_size // (patch_size * (2 ** num_downsamples))
        upsample_factor = img_size // final_size
        
        self.recon_conv = nn.Conv2d(in_dim, 3 * (upsample_factor**2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upsample_factor)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: B, C, H, W (32x32)
        x = self.patch_embed(x)  # B, 16, 16, C'
        for stage in self.stages:
            x = stage(x)  # After both stages: B, 8, 8, C
        B, H, W, C = x.shape
        x = self.norm(x.view(B, H*W, C)).view(B, H, W, C)
        # prepare for conv: B, C, H, W
        x = x.permute(0,3,1,2).contiguous()
        x = self.recon_conv(x)  # B, 3*16, 8, 8
        x = self.pixel_shuffle(x)  # B, 3, 32, 32
        x = torch.sigmoid(x)  # constrain to [0,1]
        return x

# -------------------------
# Training utilities
# -------------------------
def compute_psnr(pred, target, eps=1e-10):
    mse = torch.mean((pred - target) ** 2, dim=[1,2,3])
    psnr = 10.0 * torch.log10(1.0 / (mse + eps))
    return psnr

def save_checkpoint(state, is_best, save_dir='checkpoints', name='checkpoint.pth'):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    torch.save(state, path)
    if is_best:
        torch.save(state, os.path.join(save_dir, 'best.pth'))

def save_sample_images(model, val_loader, device, noise_sigma, epoch, save_dir='sample_images'):
    """Save sample denoising results"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Get one batch
    images, _ = next(iter(val_loader))
    images = images[:8].to(device)  # Take 8 samples
    
    sigma = noise_sigma / 255.0
    noisy = images + torch.randn_like(images) * sigma
    noisy = noisy.clamp(0.0, 1.0)
    
    with torch.no_grad():
        denoised = model(noisy)
    
    # Create comparison grid
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    
    for i in range(8):
        # Original
        img_clean = images[i].cpu().permute(1, 2, 0).numpy()
        axes[0, i].imshow(img_clean)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Clean', fontsize=10)
        
        # Noisy
        img_noisy = noisy[i].cpu().permute(1, 2, 0).numpy()
        axes[1, i].imshow(img_noisy)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Noisy', fontsize=10)
        
        # Denoised
        img_denoised = denoised[i].cpu().permute(1, 2, 0).numpy()
        axes[2, i].imshow(img_denoised)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Denoised', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch+1}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved sample images to {save_dir}/epoch_{epoch+1}.png")

# -------------------------
# Main: data, denoising training loop
# -------------------------
def main():
    # ===== CONFIGURATION =====
    img_size = 64  # Using 64x64 for STL10
    dataset_type = 'stl10'  # STL10: 13,000 images, 96x96 downsampled to 64x64
    
    epochs = 20
    batch_size = 64  # Good batch size for 64x64 images
    lr = 3e-4
    weight_decay = 1e-6
    warmup_epochs = 1
    num_workers = 4
    noise_sigma = 25.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    print(f"Training on STL10 dataset with {img_size}x{img_size} images")

    # Adjust model for 64x64 images
    model = TinySwinDenoiser(
        embed_dim=64,           
        depths=[2, 2, 6],       # 3 stages for 64x64
        num_heads=[2, 4, 8],
        window_size=4,          
        patch_size=2,           
        img_size=img_size
    )
    
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Data transforms for 64x64
    transform_train = T.Compose([
        T.Resize(64),
        T.RandomCrop(64, padding=8),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    transform_test = T.Compose([
        T.Resize(64),
        T.CenterCrop(64),
        T.ToTensor(),
    ])

    # Load STL10 dataset (easier download, no Google Drive issues)
    print("Loading STL10 dataset (will download if not present)...")
    train_set = torchvision.datasets.STL10(
        root='./data', 
        split='train',  # 5,000 labeled images
        download=True, 
        transform=transform_train
    )
    # Also use unlabeled data for more training samples
    unlabeled_set = torchvision.datasets.STL10(
        root='./data',
        split='unlabeled',  # 100,000 unlabeled images
        download=True,
        transform=transform_train
    )
    # Combine train and unlabeled (take subset of unlabeled to keep it manageable)
    unlabeled_subset = torch.utils.data.Subset(unlabeled_set, range(20000))  # Use 20k unlabeled
    train_set = torch.utils.data.ConcatDataset([train_set, unlabeled_subset])
    
    val_set = torchvision.datasets.STL10(
        root='./data', 
        split='test',  # 8,000 images
        download=True, 
        transform=transform_test
    )
    print(f"Loaded STL10: {len(train_set)} train, {len(val_set)} test images")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    def lr_lambda(current_step):
        warmup_steps = warmup_epochs * len(train_loader)
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))

    best_psnr = 0.0
    global_step = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        for i, (images, _) in pbar:
            images = images.to(device, non_blocking=True)
            sigma = noise_sigma / 255.0
            noisy = images + torch.randn_like(images) * sigma
            noisy = noisy.clamp(0.0, 1.0)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                outputs = model(noisy)
                loss = criterion(outputs, images)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': f'{running_loss/((i+1)*images.size(0)):.6f}'})

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Train epoch {epoch+1}: loss={epoch_loss:.6f}")

        model.eval()
        total_mse = 0.0
        psnr_list = []
        n_samples = 0
        with torch.no_grad():
            for images, _ in tqdm(val_loader, desc='Val'):
                images = images.to(device)
                sigma = noise_sigma / 255.0
                noisy = images + torch.randn_like(images) * sigma
                noisy = noisy.clamp(0.0, 1.0)
                outputs = model(noisy)
                mse_per_image = torch.mean((outputs - images)**2, dim=[1,2,3])
                psnrs = 10.0 * torch.log10(1.0 / (mse_per_image + 1e-10))
                psnr_list.append(psnrs.cpu().numpy())
                total_mse += torch.sum(mse_per_image).item()
                n_samples += images.size(0)

        psnr_all = np.concatenate(psnr_list, axis=0)
        mean_psnr = float(np.mean(psnr_all))
        mean_mse = total_mse / n_samples
        print(f"Val epoch {epoch+1}: MSE={mean_mse:.6e}, PSNR={mean_psnr:.3f} dB")

        is_best = mean_psnr > best_psnr
        best_psnr = max(mean_psnr, best_psnr)
        save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict(), 'best_psnr': best_psnr, 'optimizer': optimizer.state_dict()}, is_best)
        
        # Save sample denoising results
        if (epoch + 1) % 5 == 0 or epoch == 0:  # Save every 5 epochs and first epoch
            save_sample_images(model, val_loader, device, noise_sigma, epoch)

    print("Training complete. Best val PSNR: {:.3f} dB".format(best_psnr))


if __name__ == '__main__':
    main()
