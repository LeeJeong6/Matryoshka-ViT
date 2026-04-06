"""
Matryoshka Sub-Dimension ViT (MSD-ViT)
=======================================

표준 MRL은 출력 head에만 nested supervision을 적용한다.
MSD-ViT는 backbone 내부 weight(Q/K/V, MLP, patch embed 등) 전체에
matryoshka 속성을 부여한다:

  weight[:dim, :dim] 이 독립적으로 유효한 dim-dim sub-network를 구성하도록
  훈련 중 full-dim forward와 sub-dim forward를 동시에 실행하고
  양쪽 loss의 gradient를 공유 weight 영역에 흘린다.

Usage (single node, 6 GPUs):
  torchrun --nproc_per_node=6 vit_matryoshka_subdim.py \\
      --data /raid/Datasets/imagenet/ \\
      --sub-dims 96 192 384 768 \\
      --sub-heads 3 6 6 12 \\
      --epochs 90 --batch-size 128 --lr 1e-3

  # Single GPU / debug:
  python vit_matryoshka_subdim.py \\
      --data /raid/Datasets/imagenet/ \\
      --sub-dims 96 192 384 768 \\
      --sub-heads 3 6 6 12 \\
      --epochs 1 --batch-size 64 --no-ddp
"""

import argparse, math, os
from contextlib import suppress

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

import timm


# ──────────────────────────────────────────────────────────────
# 1. MSDViT  (Matryoshka Sub-Dimension ViT)
# ──────────────────────────────────────────────────────────────

class MSDViT(nn.Module):
    """
    timm VisionTransformer backbone의 weight를 공유하면서
    full-dim 경로와 sub-dim 경로를 동시에 forward하는 MSD-ViT.

    forward_at_dim(x, dim, num_heads):
      timm 모듈을 교체하지 않고 backbone 파라미터에
      F.conv2d / F.linear / F.layer_norm 으로 직접 접근.
      weight slice는 view이므로 autograd 유지 → gradient 공유.

    forward(x):
      sub-dim 경로(ascending) + full-dim 경로를 모두 실행해
      MatryoshkaLoss가 기대하는 list[Tensor] 를 반환.
    """

    def __init__(
        self,
        backbone: nn.Module,     # timm ViT, num_classes=0
        sub_dims: list,          # e.g. [96, 192, 384, 768]  ascending
        sub_heads: list,         # e.g. [3, 6, 6, 12]  num_heads per dim
        num_classes: int = 1000,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        assert sub_dims == sorted(sub_dims), "sub_dims must be ascending"
        assert len(sub_dims) == len(sub_heads)
        for d, nh in zip(sub_dims, sub_heads):
            assert d % nh == 0, f"dim={d} not divisible by num_heads={nh}"

        self.backbone  = backbone
        self.full_dim  = backbone.embed_dim
        self.sub_dims  = sub_dims
        self.sub_heads = sub_heads
        self.mlp_ratio = mlp_ratio

        assert sub_dims[-1] == self.full_dim, \
            f"sub_dims[-1]={sub_dims[-1]} must equal backbone embed_dim={self.full_dim}"

        self.head = MatryoshkaHead(self.full_dim, num_classes, sub_dims)

    # ── core: sub-dim forward ──────────────────────────────────

    def forward_at_dim(self, x: torch.Tensor, dim: int, num_heads: int) -> torch.Tensor:
        """
        backbone의 weight를 dim×dim sub-block으로 slice해 forward.
        slice는 view → gradient가 원본 파라미터로 전달됨.

        Returns: CLS token feature [B, dim]
        """
        bb      = self.backbone
        D       = self.full_dim
        mlp_dim = int(dim * self.mlp_ratio)
        B       = x.shape[0]

        # ── patch embedding ──────────────────────────────────
        pe = bb.patch_embed.proj
        x = F.conv2d(
            x,
            pe.weight[:dim],
            pe.bias[:dim] if pe.bias is not None else None,
            stride=pe.stride,
            padding=pe.padding,
        )                                                    # [B, dim, 14, 14]
        x = x.flatten(2).transpose(1, 2)                    # [B, 196, dim]

        # ── cls token + positional embedding ─────────────────
        cls = bb.cls_token[:, :, :dim].expand(B, -1, -1)    # [B, 1, dim]
        x   = torch.cat([cls, x], dim=1)                    # [B, 197, dim]
        x   = bb.pos_drop(x + bb.pos_embed[:, :, :dim])

        # ── transformer blocks ────────────────────────────────
        for blk in bb.blocks:
            # --- Attention ---
            h = F.layer_norm(x, (dim,),
                             blk.norm1.weight[:dim],
                             blk.norm1.bias[:dim])

            qkv_w = blk.attn.qkv.weight   # [3D, D]
            qkv_b = blk.attn.qkv.bias     # [3D] or None

            # view slices: gradient flows back through SliceBackward
            w_q   = qkv_w[0*D : 1*D, :D][:dim, :dim]
            w_k   = qkv_w[1*D : 2*D, :D][:dim, :dim]
            w_v   = qkv_w[2*D : 3*D, :D][:dim, :dim]
            w_qkv = torch.cat([w_q, w_k, w_v], dim=0)       # [3*dim, dim]

            if qkv_b is not None:
                b_qkv = torch.cat([
                    qkv_b[0*D : 1*D][:dim],
                    qkv_b[1*D : 2*D][:dim],
                    qkv_b[2*D : 3*D][:dim],
                ])
            else:
                b_qkv = None

            N        = x.shape[1]                            # 197
            head_dim = dim // num_heads
            scale    = head_dim ** -0.5                      # 반드시 재계산

            qkv_out = F.linear(h, w_qkv, b_qkv)             # [B, N, 3*dim]
            qkv_out = (qkv_out
                       .reshape(B, N, 3, num_heads, head_dim)
                       .permute(2, 0, 3, 1, 4))              # [3, B, nh, N, hd]
            q, k, v = qkv_out[0], qkv_out[1], qkv_out[2]

            attn = (q @ k.transpose(-2, -1)) * scale         # [B, nh, N, N]
            attn = attn.softmax(dim=-1)
            attn = blk.attn.attn_drop(attn)

            h = (attn @ v).transpose(1, 2).reshape(B, N, dim)
            h = F.linear(
                h,
                blk.attn.proj.weight[:dim, :dim],
                blk.attn.proj.bias[:dim] if blk.attn.proj.bias is not None else None,
            )
            h = blk.attn.proj_drop(h)
            x = x + blk.drop_path(h)

            # --- MLP ---
            h = F.layer_norm(x, (dim,),
                             blk.norm2.weight[:dim],
                             blk.norm2.bias[:dim])
            h = F.linear(h,
                         blk.mlp.fc1.weight[:mlp_dim, :dim],
                         blk.mlp.fc1.bias[:mlp_dim])
            h = blk.mlp.act(h)
            h = blk.mlp.drop(h)
            h = F.linear(h,
                         blk.mlp.fc2.weight[:dim, :mlp_dim],
                         blk.mlp.fc2.bias[:dim])
            h = blk.mlp.drop(h)
            x = x + blk.drop_path(h)

        # ── final norm + CLS ──────────────────────────────────
        x = F.layer_norm(x, (dim,),
                         bb.norm.weight[:dim],
                         bb.norm.bias[:dim])
        return x[:, 0]   # [B, dim]

    # ── full forward (MatryoshkaLoss 호환) ────────────────────

    def forward(self, x: torch.Tensor) -> list:
        """
        ascending sub_dims 순서로 logit list 반환.
        [logit_dim0, logit_dim1, ..., logit_full]
        → MatryoshkaLoss / evaluate 와 동일한 인터페이스.
        """
        # Full-dim: timm 최적화 경로
        full_feat = self.backbone(x)            # [B, D]

        all_logits = []
        # Sub-dim forwards (sub_dims[:-1] = full_dim 제외)
        for dim, nh in zip(self.sub_dims[:-1], self.sub_heads[:-1]):
            feat = self.forward_at_dim(x, dim, nh)             # [B, dim]
            all_logits.append(self.head.forward_scale(feat, dim))

        # Full-dim logit (마지막 — sub_dims[-1] == full_dim)
        all_logits.append(self.head.forward_scale(full_feat, self.full_dim))
        return all_logits

    def forward_single(self, x: torch.Tensor, scale_idx: int = -1) -> torch.Tensor:
        """Inference: 특정 scale 하나만 반환."""
        dim = self.sub_dims[scale_idx]
        nh  = self.sub_heads[scale_idx]
        if dim == self.full_dim:
            feat = self.backbone(x)
        else:
            feat = self.forward_at_dim(x, dim, nh)
        return self.head.forward_scale(feat, dim)


# ──────────────────────────────────────────────────────────────
# 2. Factory
# ──────────────────────────────────────────────────────────────

def build_msd_vit(
    model_name: str,
    sub_dims: list,
    sub_heads: list,
    num_classes: int      = 1000,
    pretrained: bool      = True,
    drop_path_rate: float = 0.1,
    mlp_ratio: float      = 4.0,
) -> MSDViT:
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
        drop_path_rate=drop_path_rate,
    )
    return MSDViT(backbone, sub_dims, sub_heads, num_classes, mlp_ratio)


# ──────────────────────────────────────────────────────────────
# 3. Verify sub-dim numerical correctness
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def verify_subdim(
    model: MSDViT,
    device: str = "cpu",
    tol: float  = 5e-2,
) -> dict:
    """
    forward_at_dim(x, dim, nh) 결과와
    동일 weight로 만든 독립 VisionTransformer(embed_dim=dim) 출력을 비교.

    수치가 일치 → slicing 구현 정확.
    훈련 후 sub-dim accuracy 향상은 별도로 evaluate()로 확인.
    """
    from timm.models.vision_transformer import VisionTransformer

    model.eval().to(device)
    x  = torch.randn(2, 3, 224, 224, device=device)
    bb = model.backbone
    D  = model.full_dim

    print(f"\n{'─'*55}")
    print(f"[verify_subdim]  tol={tol}")

    results = {}
    for dim, nh in zip(model.sub_dims[:-1], model.sub_heads[:-1]):
        mlp_dim  = int(dim * model.mlp_ratio)
        depth    = len(bb.blocks)
        qkv_bias = bb.blocks[0].attn.qkv.bias is not None

        # 독립 dim-dim ViT 생성 후 weight copy (clone)
        spliced = VisionTransformer(
            img_size=224, patch_size=16, in_chans=3, num_classes=0,
            embed_dim=dim, depth=depth, num_heads=nh,
            mlp_ratio=model.mlp_ratio, qkv_bias=qkv_bias,
        ).to(device)

        spliced.cls_token.copy_(bb.cls_token[:, :, :dim])
        spliced.pos_embed.copy_(bb.pos_embed[:, :, :dim])
        spliced.patch_embed.proj.weight.copy_(bb.patch_embed.proj.weight[:dim])
        if bb.patch_embed.proj.bias is not None:
            spliced.patch_embed.proj.bias.copy_(bb.patch_embed.proj.bias[:dim])
        spliced.norm.weight.copy_(bb.norm.weight[:dim])
        spliced.norm.bias.copy_(bb.norm.bias[:dim])

        for bf, bs in zip(bb.blocks, spliced.blocks):
            qw = bf.attn.qkv.weight
            qb = bf.attn.qkv.bias
            bs.attn.qkv.weight.copy_(torch.cat([
                qw[0*D:1*D, :D][:dim, :dim],
                qw[1*D:2*D, :D][:dim, :dim],
                qw[2*D:3*D, :D][:dim, :dim],
            ], dim=0))
            if qb is not None:
                bs.attn.qkv.bias.copy_(torch.cat([
                    qb[0*D:1*D][:dim],
                    qb[1*D:2*D][:dim],
                    qb[2*D:3*D][:dim],
                ]))
            bs.attn.proj.weight.copy_(bf.attn.proj.weight[:dim, :dim])
            if bf.attn.proj.bias is not None:
                bs.attn.proj.bias.copy_(bf.attn.proj.bias[:dim])
            bs.norm1.weight.copy_(bf.norm1.weight[:dim])
            bs.norm1.bias.copy_(bf.norm1.bias[:dim])
            bs.norm2.weight.copy_(bf.norm2.weight[:dim])
            bs.norm2.bias.copy_(bf.norm2.bias[:dim])
            bs.mlp.fc1.weight.copy_(bf.mlp.fc1.weight[:mlp_dim, :dim])
            bs.mlp.fc1.bias.copy_(bf.mlp.fc1.bias[:mlp_dim])
            bs.mlp.fc2.weight.copy_(bf.mlp.fc2.weight[:dim, :mlp_dim])
            bs.mlp.fc2.bias.copy_(bf.mlp.fc2.bias[:dim])

        spliced.eval()
        feat_ref = spliced(x)
        feat_msd = model.forward_at_dim(x, dim, nh)

        max_diff = (feat_msd - feat_ref).abs().max().item()
        status   = "OK  " if max_diff < tol else "WARN"
        print(f"  dim={dim:4d}  num_heads={nh:2d}  max_diff={max_diff:.4f}  [{status}]")
        results[dim] = max_diff

    print(f"{'─'*55}\n")
    return results


# ──────────────────────────────────────────────────────────────
# 4. MatryoshkaHead  (shared weight matrix)
# ──────────────────────────────────────────────────────────────

class MatryoshkaHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, mrl_dims: list):
        super().__init__()
        assert mrl_dims == sorted(mrl_dims)
        assert mrl_dims[-1] <= in_features
        self.mrl_dims = mrl_dims
        self.weight   = nn.Parameter(torch.empty(num_classes, in_features))
        self.bias     = nn.Parameter(torch.zeros(num_classes))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward_scale(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        return x[..., :dim] @ self.weight[:, :dim].T + self.bias

    def forward(self, x: torch.Tensor) -> list:
        return [self.forward_scale(x, d) for d in self.mrl_dims]


# ──────────────────────────────────────────────────────────────
# 5. Loss
# ──────────────────────────────────────────────────────────────

class LabelSmoothCE(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = torch.log_softmax(logits, dim=-1)
        nll      = -log_prob.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth   = -log_prob.mean(dim=-1)
        return ((1 - self.smoothing) * nll + self.smoothing * smooth).mean()


class MatryoshkaLoss(nn.Module):
    def __init__(self, mrl_dims: list, weights: list = None, smoothing: float = 0.1):
        super().__init__()
        K       = len(mrl_dims)
        weights = weights or [1.0 / K] * K
        assert len(weights) == K
        self.register_buffer("weights", torch.tensor(weights))
        self.criterion = LabelSmoothCE(smoothing)

    def forward(self, multi_logits: list, targets: torch.Tensor) -> torch.Tensor:
        return sum(w * self.criterion(logits, targets)
                   for w, logits in zip(self.weights, multi_logits))


# ──────────────────────────────────────────────────────────────
# 6. DataLoader
# ──────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_loaders(args, rank: int, world_size: int):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(args.data, "train"), train_transform)
    val_dataset   = datasets.ImageFolder(os.path.join(args.data, "val"),   val_transform)

    train_sampler = DistributedSampler(train_dataset, world_size, rank, shuffle=True) \
                    if world_size > 1 else None
    val_sampler   = DistributedSampler(val_dataset,   world_size, rank, shuffle=False) \
                    if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2,
        sampler=val_sampler, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    return train_loader, val_loader, train_sampler


# ──────────────────────────────────────────────────────────────
# 7. DDP helpers
# ──────────────────────────────────────────────────────────────

def setup_ddp():
    rank       = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def is_main(rank): return rank == 0

def cleanup_ddp(world_size):
    if world_size > 1:
        dist.destroy_process_group()


# ──────────────────────────────────────────────────────────────
# 8. Accuracy
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    B    = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum() / B * 100 for k in topk]


# ──────────────────────────────────────────────────────────────
# 9. Train / Eval
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    epoch, args, rank):
    model.train()
    amp_ctx  = autocast if args.amp else suppress
    loss_sum = 0.0

    for i, (images, targets) in enumerate(loader):
        images  = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        with amp_ctx():
            multi_logits = model(images)
            loss         = criterion(multi_logits, targets)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        loss_sum += loss.item()
        if is_main(rank) and i % args.log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  [{epoch}][{i}/{len(loader)}]  loss={loss.item():.4f}  lr={lr:.2e}")

    return loss_sum / len(loader)


@torch.no_grad()
def evaluate(model, loader, sub_dims, rank, world_size, amp_ctx):
    model.eval()
    K    = len(sub_dims)
    sums = [[0.0, 0.0] for _ in range(K)]
    n    = 0

    for images, targets in loader:
        images  = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        B       = images.size(0)
        n      += B

        with amp_ctx():
            multi_logits = model(images)

        for i, logits in enumerate(multi_logits):
            a1, a5 = accuracy(logits, targets, topk=(1, 5))
            sums[i][0] += a1.item() * B
            sums[i][1] += a5.item() * B

    results = {}
    for i, d in enumerate(sub_dims):
        t1 = torch.tensor(sums[i][0] / n, device="cuda")
        t5 = torch.tensor(sums[i][1] / n, device="cuda")
        if world_size > 1:
            dist.all_reduce(t1, op=dist.ReduceOp.AVG)
            dist.all_reduce(t5, op=dist.ReduceOp.AVG)
        results[d] = (t1.item(), t5.item())
    return results


# ──────────────────────────────────────────────────────────────
# 10. Scheduler
# ──────────────────────────────────────────────────────────────

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        ep = self.last_epoch
        if ep < self.warmup_epochs:
            factor = (ep + 1) / max(self.warmup_epochs, 1)
        else:
            progress = (ep - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            factor   = self.min_lr / self.base_lrs[0] + \
                       (1 - self.min_lr / self.base_lrs[0]) * \
                       0.5 * (1 + math.cos(math.pi * progress))
        return [base_lr * factor for base_lr in self.base_lrs]


# ──────────────────────────────────────────────────────────────
# 11. Args
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MSD-ViT: Matryoshka Sub-Dimension ViT")
    # data / model
    p.add_argument("--data",            default="/mnt/hdd_10tb_sda/IMAGENET1k/")
    p.add_argument("--model",           default="vit_base_patch16_224")
    p.add_argument("--num-classes",     type=int,   default=1000)
    p.add_argument("--pretrained",      action="store_true", default=True)
    p.add_argument("--no-pretrained",   dest="pretrained", action="store_false")
    # sub-dim config
    p.add_argument("--sub-dims",   type=int, nargs="+", default=[96, 192, 384, 768],
                   help="Sub-dim sizes ascending; last must equal backbone embed_dim")
    p.add_argument("--sub-heads",  type=int, nargs="+", default=[3, 6, 6, 12],
                   help="num_heads per sub-dim; dim %% num_heads == 0 required")
    p.add_argument("--sub-weights", type=float, nargs="+", default=None,
                   help="Loss weights per sub-dim (uniform if None)")
    p.add_argument("--drop-path-rate", type=float, default=0.1)
    p.add_argument("--mlp-ratio",      type=float, default=4.0)
    # training
    p.add_argument("--epochs",          type=int,   default=90)
    p.add_argument("--batch-size",      type=int,   default=64)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--weight-decay",    type=float, default=0.05)
    p.add_argument("--warmup-epochs",   type=int,   default=5)
    p.add_argument("--min-lr",          type=float, default=1e-6)
    p.add_argument("--clip-grad",       type=float, default=1.0)
    p.add_argument("--smoothing",       type=float, default=0.1)
    p.add_argument("--workers",         type=int,   default=8)
    p.add_argument("--amp",             action="store_true", default=True)
    p.add_argument("--no-amp",          dest="amp", action="store_false")
    p.add_argument("--no-ddp",          action="store_true")
    # logging / saving
    p.add_argument("--output",          default="./output_msd")
    p.add_argument("--log-interval",    type=int,   default=50)
    p.add_argument("--eval-every",      type=int,   default=1)
    p.add_argument("--resume",          default="")
    # verification
    p.add_argument("--verify-after-train",    action="store_true", default=True)
    p.add_argument("--no-verify-after-train", dest="verify_after_train",
                   action="store_false")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# 12. Main
# ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(42)

    rank, local_rank, world_size = setup_ddp()
    if args.no_ddp:
        world_size = 1
    device  = torch.device(f"cuda:{local_rank}")
    amp_ctx = autocast if args.amp else suppress

    if is_main(rank):
        os.makedirs(args.output, exist_ok=True)
        print(f"Model     : {args.model}")
        print(f"Sub-dims  : {args.sub_dims}")
        print(f"Sub-heads : {args.sub_heads}")
        print(f"GPUs      : {world_size}")

    # ── Model ──
    model = build_msd_vit(
        args.model, args.sub_dims, args.sub_heads,
        args.num_classes, args.pretrained,
        drop_path_rate=args.drop_path_rate,
        mlp_ratio=args.mlp_ratio,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ── Data ──
    train_loader, val_loader, train_sampler = build_loaders(args, rank, world_size)

    # ── Loss / Optimizer / Scheduler ──
    criterion = MatryoshkaLoss(args.sub_dims, args.sub_weights, args.smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(
        optimizer, args.warmup_epochs, args.epochs, args.min_lr,
    )
    scaler = GradScaler() if args.amp else None

    # ── Resume ──
    start_epoch = 0
    best_top1   = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        m    = model.module if world_size > 1 else model
        m.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_top1   = ckpt.get("best_top1", 0.0)
        if is_main(rank):
            print(f"Resumed from epoch {start_epoch}")

    # ══════════════════════════════
    # Training loop
    # ══════════════════════════════
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, args, rank,
        )
        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            results = evaluate(
                model.module if world_size > 1 else model,
                val_loader, args.sub_dims, rank, world_size, amp_ctx,
            )

            if is_main(rank):
                print(f"\n{'─'*58}")
                print(f"Epoch {epoch:3d}  loss={train_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")
                print(f"{'dim':>8}  {'Top-1':>7}  {'Top-5':>7}")
                for d, (t1, t5) in results.items():
                    tag = " ◀ full" if d == args.sub_dims[-1] else ""
                    print(f"{d:>8}  {t1:>7.2f}  {t5:>7.2f}{tag}")
                print(f"{'─'*58}\n")

                full_top1 = results[args.sub_dims[-1]][0]
                is_best   = full_top1 > best_top1
                best_top1 = max(full_top1, best_top1)

                m    = model.module if world_size > 1 else model
                ckpt = {
                    "epoch":     epoch,
                    "model":     m.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_top1": best_top1,
                    "sub_dims":  args.sub_dims,
                    "sub_heads": args.sub_heads,
                }
                torch.save(ckpt, os.path.join(args.output, "last.pth"))
                if is_best:
                    torch.save(ckpt, os.path.join(args.output, "best.pth"))
                    print(f"  ★ Best Top-1 (full dim): {best_top1:.2f}")

    # ── Post-training verification ──
    if is_main(rank) and args.verify_after_train:
        m = model.module if world_size > 1 else model
        verify_subdim(m, device=str(device))

    cleanup_ddp(world_size)
    if is_main(rank):
        print("Done.")


if __name__ == "__main__":
    main()
