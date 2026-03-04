# make_notebook.py
import json, textwrap
from pathlib import Path

def md(s):
    return {"cell_type":"markdown","metadata":{},"source":textwrap.dedent(s).strip()+"\n"}

def code(s):
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
            "source":textwrap.dedent(s).strip()+"\n"}

nb = {
  "cells": [],
  "metadata": {
    "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
    "language_info": {"name":"python","version":"3.x"}
  },
  "nbformat": 4,
  "nbformat_minor": 5
}

nb["cells"].append(md("""
# Jaguar Re-ID — EVA-02 memory-safe SOTA training (PK + accumulation + checkpointing)

Этот ноутбук применяет **все memory-safe улучшения**:
- img_size=384, embed_dim=512
- P×K sampler + gradient accumulation (виртуальный большой PK)
- gradient checkpointing (если доступно)
- AMP/BF16 + TF32
- progressive unfreeze (1 эпоха head-only, затем last N blocks)
- inference: flip TTA + weighted AQE
- пишет `submission.csv`

Запускать сверху вниз на Kaggle.
"""))

nb["cells"].append(code(r"""
# =========================
# 0. Setup
# =========================
import os, gc, math, time, json, random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception:
    !pip -q install albumentations==1.4.24 opencv-python-headless timm scikit-learn
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

import cv2
import timm
from sklearn.model_selection import GroupKFold

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

# =========================
# 1. Config
# =========================
@dataclass
class CFG:
    data_root: str = "/kaggle/input/jaguar-re-id"
    out_dir: str = "./artifacts_eva02_mem"

    backbone: str = "eva02_large_patch14_448.mim_in22k_ft_in22k_in1k"
    img_size: int = 384          # ✅ reduced image
    embed_dim: int = 512         # ✅ reduced embedding
    gem_p_init: float = 3.0

    seed: int = 42
    epochs: int = 6
    num_workers: int = 4

    # PK effective batch
    P: int = 12
    K: int = 4

    # micro-batch + accumulation (virtual PK)
    micro_batch: int = 8
    accum_steps: int = 6         # micro_batch*accum_steps == P*K

    lr_head: float = 3e-4
    lr_backbone: float = 3e-5
    weight_decay: float = 0.05
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0

    # progressive unfreeze
    freeze_epochs: int = 1
    unfreeze_last_blocks: int = 8

    # memory saver
    grad_checkpointing: bool = True

    # ArcFace
    arc_s: float = 32.0
    arc_m: float = 0.40

    # alpha-mask crop
    use_mask_crop: bool = True
    mask_min_area: int = 25
    strong_aug: bool = True

    # inference
    tta_flip: bool = True
    post_aqe: bool = True
    aqe_k: int = 3
    aqe_alpha: float = 2.0

cfg = CFG()
os.makedirs(cfg.out_dir, exist_ok=True)
assert cfg.micro_batch * cfg.accum_steps == cfg.P * cfg.K, "micro_batch*accum_steps must equal P*K"
print(json.dumps(asdict(cfg), indent=2))

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True

seed_everything(cfg.seed)

# =========================
# 2. Load data
# =========================
DATA_ROOT = Path(cfg.data_root)
train_df = pd.read_csv(DATA_ROOT / "train.csv")
test_df  = pd.read_csv(DATA_ROOT / "test.csv")

img_candidates = ["image","file","file_name","filename","img","path"]
id_candidates  = ["individual_id","id","label","identity","jaguar_id","class"]
img_col = next((c for c in img_candidates if c in train_df.columns), None)
id_col  = next((c for c in id_candidates if c in train_df.columns), None)
assert img_col and id_col, train_df.columns.tolist()
print("Detected:", img_col, id_col)

train_df["label_str"] = train_df[id_col].astype(str)
classes = sorted(train_df["label_str"].unique())
cls2idx = {c:i for i,c in enumerate(classes)}
train_df["target"] = train_df["label_str"].map(cls2idx).astype(int)
num_classes = len(classes)
print("num_classes:", num_classes)

# =========================
# 3. Transforms + mask crop
# =========================
def build_transforms(img_size: int, strong: bool):
    if strong:
        train_tf = A.Compose([
            A.RandomResizedCrop(img_size, img_size, scale=(0.78, 1.0), ratio=(0.80, 1.25), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.08, rotate_limit=8,
                               border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.18, hue=0.05, p=0.7),
            A.RandomBrightnessContrast(p=0.25),
            A.GaussNoise(p=0.15),
            A.CoarseDropout(max_holes=2, max_height=img_size//7, max_width=img_size//7,
                            fill_value=127, p=0.20),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        train_tf = A.Compose([A.Resize(img_size,img_size), A.HorizontalFlip(p=0.5), A.Normalize(), ToTensorV2()])
    val_tf = A.Compose([A.Resize(img_size,img_size), A.Normalize(), ToTensorV2()])
    return train_tf, val_tf

train_tf, val_tf = build_transforms(cfg.img_size, cfg.strong_aug)

def read_image_rgba(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        a = np.full((img.shape[0], img.shape[1], 1), 255, dtype=img.dtype)
        img = np.concatenate([img, a], axis=2)
    return img  # BGRA

def mask_crop_and_pad(bgra: np.ndarray, min_area: int = 25) -> np.ndarray:
    bgr = bgra[..., :3]
    a   = bgra[..., 3]
    mask = (a > 10).astype(np.uint8)
    if mask.sum() < min_area:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    ys, xs = np.where(mask > 0)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    h, w = bgr.shape[:2]
    pad = int(0.05 * max(y1-y0+1, x1-x0+1))
    y0 = max(0, y0-pad); y1 = min(h-1, y1+pad)
    x0 = max(0, x0-pad); x1 = min(w-1, x1+pad)

    crop_bgr = bgr[y0:y1+1, x0:x1+1]
    crop_a   = a[y0:y1+1, x0:x1+1].astype(np.float32) / 255.0

    neutral = np.full_like(crop_bgr, 127)
    crop_bgr = (crop_bgr*crop_a[...,None] + neutral*(1.0-crop_a[...,None])).astype(np.uint8)

    ch, cw = crop_bgr.shape[:2]
    s = max(ch, cw)
    out = np.full((s,s,3), 127, dtype=np.uint8)
    yoff = (s-ch)//2; xoff = (s-cw)//2
    out[yoff:yoff+ch, xoff:xoff+cw] = crop_bgr
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

# =========================
# 4. Dataset + PK sampler
# =========================
class JaguarDataset(Dataset):
    def __init__(self, df, img_dir, tfm):
        self.df=df.reset_index(drop=True)
        self.img_dir=img_dir
        self.tfm=tfm
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row=self.df.iloc[idx]
        name=row[img_col]
        bgra=read_image_rgba(str(self.img_dir/name))
        if cfg.use_mask_crop:
            img=mask_crop_and_pad(bgra, cfg.mask_min_area)
        else:
            img=cv2.cvtColor(bgra[...,:3], cv2.COLOR_BGR2RGB)
        x=self.tfm(image=img)["image"]
        return x, int(row["target"]), name

class PKBatchSampler(Sampler):
    def __init__(self, labels, P, K, batches_per_epoch=None):
        self.labels=np.asarray(labels)
        self.P=P; self.K=K
        self.idxs_by_label={int(y): np.where(self.labels==y)[0] for y in np.unique(self.labels)}
        self.unique_labels=np.array(sorted(self.idxs_by_label.keys()))
        self.batches_per_epoch = batches_per_epoch or max(1, len(self.labels)//(P*K))
    def __len__(self): return self.batches_per_epoch
    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            chosen=np.random.choice(self.unique_labels, size=self.P, replace=(len(self.unique_labels)<self.P))
            batch=[]
            for y in chosen:
                pool=self.idxs_by_label[int(y)]
                pick=np.random.choice(pool, size=self.K, replace=(len(pool)<self.K))
                batch.extend(pick.tolist())
            yield batch

def make_split(df, n_splits=5, fold=0):
    gkf=GroupKFold(n_splits=n_splits)
    groups=df["target"].values
    tr,va=list(gkf.split(df, df["target"], groups))[fold]
    return df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)

train_split, val_split = make_split(train_df, 5, 0)

train_img_dir = DATA_ROOT/"train"/"train"
test_img_dir  = DATA_ROOT/"test"/"test"

train_ds = JaguarDataset(train_split, train_img_dir, train_tf)
val_ds   = JaguarDataset(val_split,   train_img_dir, val_tf)

train_loader = DataLoader(
    train_ds,
    batch_sampler=PKBatchSampler(train_split["target"].tolist(), cfg.P, cfg.K),
    num_workers=cfg.num_workers,
    pin_memory=True
)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)

# =========================
# 5. Model: EVA + token GeM + BNNeck + ArcFace
# =========================
class GeM(nn.Module):
    def __init__(self,p=3.0,eps=1e-6):
        super().__init__()
        self.p=nn.Parameter(torch.ones(1)*p); self.eps=eps
    def forward(self,x):
        x=x.transpose(1,2).clamp(min=self.eps).pow(self.p)
        return x.mean(dim=-1).pow(1.0/self.p)

class ArcMarginProduct(nn.Module):
    def __init__(self,in_features,out_features,s=32.0,m=0.40):
        super().__init__()
        self.s=s; self.m=m
        self.weight=nn.Parameter(torch.empty(out_features,in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m=math.cos(m); self.sin_m=math.sin(m)
        self.th=math.cos(math.pi-m); self.mm=math.sin(math.pi-m)*m
    def forward(self,emb,label):
        cosine=F.linear(emb, F.normalize(self.weight))
        sine=torch.sqrt(torch.clamp(1.0-cosine*cosine, min=1e-9))
        phi=cosine*self.cos_m - sine*self.sin_m
        phi=torch.where(cosine>self.th, phi, cosine-self.mm)
        one_hot=torch.zeros_like(cosine); one_hot.scatter_(1,label.view(-1,1),1.0)
        logits=(one_hot*phi)+((1.0-one_hot)*cosine)
        return logits*self.s

class EVAReID(nn.Module):
    def __init__(self, backbone_name, embed_dim, gem_p_init):
        super().__init__()
        self.backbone=timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool="")
        if cfg.grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)

        with torch.no_grad():
            d=torch.zeros(1,3,cfg.img_size,cfg.img_size)
            f=self.backbone.forward_features(d)
            if isinstance(f,(list,tuple)): f=f[-1]
            feat_dim=f.shape[-1]

        self.gem=GeM(gem_p_init)
        self.proj=nn.Linear(feat_dim, embed_dim)
        self.bnneck=nn.BatchNorm1d(embed_dim)
        self.bnneck.bias.requires_grad_(False)

    def forward(self,x):
        f=self.backbone.forward_features(x)
        if isinstance(f,(list,tuple)): f=f[-1]
        patch=f[:,1:,:] if f.dim()==3 and f.shape[1]>1 else f
        pooled=self.gem(patch)
        emb=self.bnneck(self.proj(pooled))
        return F.normalize(emb,p=2,dim=1)

class EVAWithArcFace(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__()
        self.base=base
        self.arc=ArcMarginProduct(cfg.embed_dim, num_classes, s=cfg.arc_s, m=cfg.arc_m)
    def forward(self,x,y=None):
        emb=self.base(x)
        if y is None: return emb
        return emb, self.arc(emb,y)

model = EVAWithArcFace(EVAReID(cfg.backbone,cfg.embed_dim,cfg.gem_p_init), num_classes).to(DEVICE)

def set_trainable(m, flag: bool):
    for p in m.parameters():
        p.requires_grad = flag

def freeze_backbone_all(base):
    set_trainable(base.backbone, False)

def progressive_unfreeze(base, n_last: int):
    if not hasattr(base.backbone, "blocks"):
        set_trainable(base.backbone, True)
        return
    blocks = base.backbone.blocks
    n = len(blocks)
    set_trainable(base.backbone, False)
    for i in range(max(0, n-n_last), n):
        set_trainable(blocks[i], True)
    if hasattr(base.backbone, "norm"):
        set_trainable(base.backbone.norm, True)

# AMP/BF16
use_bf16 = (DEVICE=="cuda") and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda" and not use_bf16))
criterion = nn.CrossEntropyLoss()

def build_optimizer():
    bb=[p for p in model.base.backbone.parameters() if p.requires_grad]
    head=[]
    for m in [model.base.gem, model.base.proj, model.base.bnneck, model.arc]:
        head += [p for p in m.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        [{"params": bb, "lr": cfg.lr_backbone, "weight_decay": cfg.weight_decay},
         {"params": head, "lr": cfg.lr_head, "weight_decay": cfg.weight_decay}]
    )

def make_scheduler(opt, total_steps: int):
    warm=int(total_steps*cfg.warmup_ratio)
    def f(step):
        if step < warm: return float(step)/max(1,warm)
        prog=(step-warm)/max(1,total_steps-warm)
        return 0.5*(1.0+math.cos(math.pi*prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)

# =========================
# 6. identity-balanced mAP
# =========================
def ap(y_true, y_score):
    order=np.argsort(-y_score); y_true=y_true[order]
    if y_true.sum()==0: return 0.0
    c=np.cumsum(y_true)
    p=c/(np.arange(len(y_true))+1)
    return float((p*y_true).sum()/y_true.sum())

def ib_map(emb, labels):
    sim=emb@emb.T; np.fill_diagonal(sim, -1e9)
    ids=np.unique(labels); per=[]
    for _id in ids:
        idxs=np.where(labels==_id)[0]
        aps=[]
        for qi in idxs:
            y=(labels==labels[qi]).astype(np.int32); y[qi]=0
            aps.append(ap(y, sim[qi]))
        per.append(float(np.mean(aps)) if len(aps) else 0.0)
    return float(np.mean(per)) if len(per) else 0.0

@torch.no_grad()
def extract_val():
    model.base.eval()
    embs=[]; ys=[]
    for x,y,_ in val_loader:
        x=x.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda"), dtype=amp_dtype):
            e=model.base(x)
        embs.append(e.float().cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(embs,0).astype(np.float32), np.concatenate(ys,0).astype(np.int64)

# =========================
# 7. Train with micro-batches + accumulation
# =========================
def train_one_epoch(opt, sch):
    model.train()
    running=0.0
    for batch_indices in train_loader.batch_sampler:
        opt.zero_grad(set_to_none=True)
        for a in range(cfg.accum_steps):
            mb=batch_indices[a*cfg.micro_batch:(a+1)*cfg.micro_batch]
            xs=[]; ys=[]
            for i in mb:
                x,y,_=train_loader.dataset[i]
                xs.append(x); ys.append(y)
            x=torch.stack(xs).to(DEVICE, non_blocking=True)
            y=torch.tensor(ys, device=DEVICE, dtype=torch.long)
            with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda"), dtype=amp_dtype):
                _,logits=model(x,y)
                loss=criterion(logits,y)/cfg.accum_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += float(loss.item())*cfg.accum_steps

        if cfg.grad_clip and cfg.grad_clip>0:
            if scaler.is_enabled(): scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        if scaler.is_enabled():
            scaler.step(opt); scaler.update()
        else:
            opt.step()
        sch.step()
    return running/max(1,len(train_loader))

best=-1.0
best_path=Path(cfg.out_dir)/"best.pt"
steps=cfg.epochs*len(train_loader)
opt=build_optimizer(); sch=make_scheduler(opt, steps)

for epoch in range(cfg.epochs):
    if epoch < cfg.freeze_epochs:
        freeze_backbone_all(model.base)
    else:
        progressive_unfreeze(model.base, cfg.unfreeze_last_blocks)

    if epoch == cfg.freeze_epochs:
        opt=build_optimizer(); sch=make_scheduler(opt, steps)

    t0=time.time()
    loss=train_one_epoch(opt, sch)
    emb,y=extract_val()
    score=ib_map(emb,y)
    print(f"Epoch {epoch+1}/{cfg.epochs} loss={loss:.4f} ib-mAP={score:.5f} time={time.time()-t0:.1f}s")
    if score>best:
        best=score
        torch.save({"cfg":asdict(cfg),"model":model.state_dict(),"classes":classes}, best_path)
        print("  saved", best_path)

print("BEST:", best)

# =========================
# 8. Inference: flip TTA + weighted AQE + submission
# =========================
ckpt=torch.load(best_path, map_location="cpu")
model.load_state_dict(ckpt["model"]); model.eval()

uq=sorted(set(test_df["query_image"])|set(test_df["gallery_image"]))

class TestDataset(Dataset):
    def __init__(self,names,img_dir,tfm):
        self.names=names; self.img_dir=img_dir; self.tfm=tfm
    def __len__(self): return len(self.names)
    def __getitem__(self,i):
        name=self.names[i]
        bgra=read_image_rgba(str(self.img_dir/name))
        if cfg.use_mask_crop: img=mask_crop_and_pad(bgra, cfg.mask_min_area)
        else: img=cv2.cvtColor(bgra[...,:3], cv2.COLOR_BGR2RGB)
        x=self.tfm(image=img)["image"]
        return x,name

test_ds=TestDataset(uq, DATA_ROOT/"test"/"test", val_tf)
test_loader=DataLoader(test_ds,batch_size=64,shuffle=False,num_workers=cfg.num_workers,pin_memory=True)

@torch.no_grad()
def extract_test():
    model.base.eval()
    embs=[]; names=[]
    for x,n in test_loader:
        x=x.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda"), dtype=amp_dtype):
            e1=model.base(x)
            if cfg.tta_flip:
                e2=model.base(torch.flip(x,dims=[3]))
                e=F.normalize((e1+e2)/2.0, p=2, dim=1)
            else:
                e=e1
        embs.append(e.float().cpu().numpy()); names.extend(list(n))
    return np.concatenate(embs,0).astype(np.float32), names

E,names=extract_test()
name2idx={n:i for i,n in enumerate(names)}

def aqe(E,k=3,alpha=2.0):
    S=E@E.T; np.fill_diagonal(S,-1e9)
    N,D=E.shape; E2=np.empty_like(E)
    for i in range(N):
        kk=min(k,N-1)
        idx=np.argpartition(-S[i], kth=kk)[:kk]
        sims=S[i,idx]
        w=np.maximum(sims,0.0)**alpha
        if w.sum()<1e-12:
            q=E[i]
        else:
            q=E[i] + (E[idx]*(w[:,None]/(w.sum()+1e-12))).sum(axis=0)
        E2[i]=(q/(np.linalg.norm(q)+1e-12)).astype(np.float32)
    return E2

Epp=aqe(E,cfg.aqe_k,cfg.aqe_alpha) if cfg.post_aqe else E
pairs=test_df[["row_id","query_image","gallery_image"]]
q=pairs["query_image"].map(name2idx).values
g=pairs["gallery_image"].map(name2idx).values
cos=(Epp[q]*Epp[g]).sum(axis=1)
scores=np.clip((cos+1.0)/2.0,0.0,1.0).astype(np.float32)

sub=pd.DataFrame({"row_id":pairs["row_id"].values,"similarity":scores}).sort_values("row_id").reset_index(drop=True)
out_path=Path(cfg.out_dir)/"submission.csv"
sub.to_csv(out_path,index=False)
print("Saved:", out_path, sub.shape)
"""))

out = Path("Jaguar_EVA02_MemorySafe_PKAccum.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Wrote:", out.resolve())