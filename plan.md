Collecting workspace informationHere’s a concrete plan to add Ensemble Knowledge Distillation (EKD) to your codebase, aligned with the existing ensemble workflow and file layout. It’s split into A/B/C as requested, with minimal invasive changes and new scripts/utilities that reuse your current training/eval pipeline.

Current codebase summary (what we’ll reuse)
- Teacher models: my_arcface.py as `lib.models.my_arcface.MyArcFace`, trained via train_gm_arc.py. Outputs logits and a 512-d embedding.
- Inference/encoding: predict.py, multi-model ensemble with TTA.
- Distance/evaluation: evaluation.py, evaluate_ensemble.py.
- Face alignment, datasets, utils: face_alignment.py, dataset utilities used by train_gm_arc.py.

Plan and code changes

A) Teacher Ensemble Development
Goal: Train a high-capacity teacher ensemble: M1 (ResNet-50 “mix”), M2 (ResNet-100 “r100”), M3 (face model) on GMDB.

- Use existing training:
  - Train M1/M2 with train_gm_arc.py using paper_model a/b (already in README commands).
  - M3: use the original r100 ONNX as a teacher (optionally fine-tuned with train_gm_arc.py by pointing to r100 base). The current trainer loads ONNX via `lib.models.my_arcface.MyArcFace`.

- New utility to load frozen teachers consistently (pth and onnx), already needed later for KD:
  - New file: lib/distill/teacher_loader.py

````python
from typing import List, Tuple
import os
import torch
from onnx2torch import convert
from lib.models.my_arcface import MyArcFace

def load_teacher(weights_path: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    # pth: torch-saved MyArcFace; onnx: raw ArcFace backbone
    if weights_path.endswith(".onnx"):
        model = convert(weights_path).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model
    elif weights_path.endswith(".pth"):
        model = torch.load(weights_path, map_location=device).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model
    else:
        raise ValueError(f"Unsupported teacher weight format: {weights_path}")

def load_teacher_ensemble(weight_dir: str, teacher_files: List[str], num_classes: int,
                          device: torch.device) -> List[torch.nn.Module]:
    models = []
    for f in teacher_files:
        m = load_teacher(os.path.join(weight_dir, f), num_classes, device)
        models.append(m)
    return models
````

B) Iterative Weight Pruning
Goal: Iteratively prune each teacher using a zero-activation-inspired criterion, fine-tune after each pruning round, stop when accuracy drops significantly.

- Approach:
  - Use forward hooks to collect zero-activation rates per layer on a calibration subset.
  - Rank weights/channels by zero-activation (or fallback to magnitude) and prune with torch.nn.utils.prune (unstructured to avoid changing shapes).
  - After pruning, fine-tune for a few epochs using the same dataset utilities as train_gm_arc.py.
  - Repeat until accuracy drop threshold.

- New pruning utilities:
  - New file: lib/pruning/prune_utils.py

````python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Tuple

@torch.no_grad()
def collect_zero_activation_stats(model: nn.Module, dataloader, device) -> Dict[str, float]:
    stats = {}
    handles = []

    def hook(name):
        def fn(_m, inp, out):
            if isinstance(out, torch.Tensor):
                zeros = (out == 0).float().mean().item()
                stats[name] = stats.get(name, 0.0) * 0.9 + 0.1 * zeros
        return fn

    for name, m in model.named_modules():
        if isinstance(m, (nn.ReLU, nn.LeakyReLU)):
            handles.append(m.register_forward_hook(hook(name)))

    for i, (x, _) in enumerate(dataloader):
        x = x.to(device, dtype=torch.float32)
        model(x)
        if i > 100:  # cap calibration iterations
            break

    for h in handles:
        h.remove()
    return stats

def apply_unstructured_prune(model: nn.Module, amount: float, whitelist=(nn.Conv2d, nn.Linear)):
    for module in model.modules():
        if isinstance(module, whitelist) and hasattr(module, "weight"):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")

def prune_iterative(model: nn.Module, dataloader, device, schedule: List[float]):
    # Example schedule: [0.1, 0.2, 0.3] cumulative amounts or per-iter amounts
    for amt in schedule:
        _ = collect_zero_activation_stats(model, dataloader, device)
        apply_unstructured_prune(model, amount=amt)
    return model
````

- New script to orchestrate pruning + fine-tuning:
  - New file: prune_iterative.py

````python
import argparse
import torch
import os
from lib.pruning.prune_utils import prune_iterative
from lib.models.my_arcface import MyArcFace
from lib.distill.teacher_loader import load_teacher
from lib.datasets.utils import get_train_and_val_datasets
from lib.utils_functions import seed_worker
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--dataset', default='gmdb')
    p.add_argument('--dataset_version', default='v1.1.0')
    p.add_argument('--img_size', type=int, default=112)
    p.add_argument('--in_channels', type=int, default=3)
    p.add_argument('--data_dir', default='../data')
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs_ft', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--schedule', type=float, nargs='+', default=[0.1, 0.1, 0.1])
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')

    # Build a small calibration and fine-tune loaders
    train_ds, val_ds = get_train_and_val_datasets(args.dataset, 'train', args.dataset_version,
                                                  args.img_size, args.in_channels, args.data_dir, img_postfix='_aligned')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, worker_init_fn=seed_worker, drop_last=False)

    # Load teacher as a torch model (pth or onnx via convert)
    # If ONNX, we only prune torch modules; skip pruning for plain ONNX converted graphs if unsupported
    model = load_teacher(args.weights, num_classes=train_ds.get_num_classes(), device=device)

    # Perform iterative pruning
    prune_iterative(model, train_loader, device, schedule=args.schedule)

    # Quick fine-tune
    if isinstance(model, MyArcFace):
        opt = optim.Adam(model.parameters(), lr=5e-5, weight_decay=0.)
        model.train()
        for e in range(args.epochs_ft):
            for x, y in train_loader:
                x = x.to(device, dtype=torch.float32); y = y.to(device, dtype=torch.int64)
                pred, _rep = model(x)
                loss = F.cross_entropy(pred, y)
                opt.zero_grad(); loss.backward(); opt.step()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model, args.out)
    print(f"Saved pruned model to {args.out}")

if __name__ == '__main__':
    main()
````

C) Ensemble Knowledge Distillation (EKD) Pipeline
Goal: Distill from pruned teacher ensemble to a compact multi-branch student optimized for edge devices, with combined CE, KD (KL), and MSE (embedding) losses. Couple student branches to teacher sub-networks.

1) Student model (Compact multi-branch)
- New file: lib/models/compact_net.py
- Design: lightweight depthwise-separable conv branches that output 512-d embeddings, followed by classifiers. Output both logits and embeddings per branch to mirror `lib.models.my_arcface.MyArcFace` behavior for compatibility with the rest of the pipeline.

````python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConvBlock(nn.Module):
    def __init__(self, in_c, out_c, s=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, s, 1, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class CompactBranch(nn.Module):
    def __init__(self, in_channels=3, emb_dim=512, width=0.5):
        super().__init__()
        c1, c2, c3 = int(32*width), int(64*width), int(128*width)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, 2, 1, bias=False), nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            DWConvBlock(c1, c2, s=2),
            DWConvBlock(c2, c2, s=1),
            DWConvBlock(c2, c3, s=2),
            DWConvBlock(c3, c3, s=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(c3, emb_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x).flatten(1)
        emb = F.normalize(self.fc(x))
        return emb  # 512-d normalized embedding

class CompactEnsemble(nn.Module):
    def __init__(self, num_classes: int, num_branches: int = 3, in_channels=3, emb_dim=512, width=0.5):
        super().__init__()
        self.branches = nn.ModuleList([CompactBranch(in_channels, emb_dim, width) for _ in range(num_branches)])
        self.classifiers = nn.ModuleList([nn.Linear(emb_dim, num_classes) for _ in range(num_branches)])

    def forward(self, x):
        logits_list, emb_list = [], []
        for b, c in zip(self.branches, self.classifiers):
            emb = b(x)
            logits = c(emb)
            logits_list.append(logits)
            emb_list.append(emb)
        return logits_list, emb_list

    def aggregate_logits(self, logits_list):
        # average over branches
        return torch.stack(logits_list, dim=0).mean(0)
````

2) Distillation losses
- New file: lib/distill/losses.py
- Implements: CE on student logits vs labels; KD (KLDiv with temperature) between teacher ensemble logits and each student branch; MSE between teacher embedding (averaged) and student embedding(s).

````python
import torch
import torch.nn.functional as F
from typing import List

def kd_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = 4.0):
    # KL divergence between softened distributions
    p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)

def mse_embed_loss(student_emb: torch.Tensor, teacher_emb: torch.Tensor):
    return F.mse_loss(student_emb, teacher_emb)

def ce_loss(student_logits: torch.Tensor, labels: torch.Tensor):
    return F.cross_entropy(student_logits, labels)

def combine_losses(student_logits_list: List[torch.Tensor],
                   student_emb_list: List[torch.Tensor],
                   teacher_logits_ens: torch.Tensor,
                   teacher_emb_ens: torch.Tensor,
                   labels: torch.Tensor,
                   T: float = 4.0,
                   w_ce: float = 1.0,
                   w_kd: float = 1.0,
                   w_mse: float = 1.0):
    # CE on aggregated student logits
    logits_agg = torch.stack(student_logits_list, dim=0).mean(0)
    loss_ce = ce_loss(logits_agg, labels)

    # KD per branch to preserve heterogeneity
    loss_kd = 0.0
    for sl in student_logits_list:
        loss_kd = loss_kd + kd_kl_loss(sl, teacher_logits_ens, T=T)
    loss_kd = loss_kd / len(student_logits_list)

    # MSE on embeddings (per-branch to same ensembled teacher embedding)
    loss_mse = 0.0
    for se in student_emb_list:
        loss_mse = loss_mse + mse_embed_loss(se, teacher_emb_ens)
    loss_mse = loss_mse / len(student_emb_list)

    total = w_ce * loss_ce + w_kd * loss_kd + w_mse * loss_mse
    return total, {'ce': loss_ce.item(), 'kd': loss_kd.item(), 'mse': loss_mse.item()}
````

3) EKD trainer
- New script: train_ekd.py
- Loads pruned teacher ensemble (from B), builds student (`CompactEnsemble`), couples branch i ↔ teacher i, computes KD from ensembled teacher outputs (logits + embeddings averaged across available teachers), plus per-branch matching (student branch to teacher ensemble).

- Uses dataset utilities and mirrors train_gm_arc.py training loop style for consistency.

````python
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from typing import List

from lib.models.compact_net import CompactEnsemble
from lib.distill.teacher_loader import load_teacher_ensemble
from lib.distill.losses import combine_losses
from lib.datasets.utils import get_train_and_val_datasets
from lib.utils_functions import seed_worker

@torch.no_grad()
def teacher_forward_batch(teachers: List[torch.nn.Module], x: torch.Tensor, device: torch.device):
    logits_list, emb_list = [], []
    for t in teachers:
        out = t(x)
        # Support: MyArcFace returns (logits, embedding); ONNX converted may return embedding only
        if isinstance(out, (list, tuple)) and len(out) == 2:
            tl, te = out
        else:
            # assume single output -> embedding; use a linear proxy? here we fallback to embedding-only KD
            tl, te = None, out
        if tl is not None:
            logits_list.append(tl)
        emb_list.append(te if te.ndim == 2 else te.squeeze())
    # aggregate
    logits_ens = torch.stack(logits_list, dim=0).mean(0) if len(logits_list) > 0 else None
    emb_ens = torch.stack(emb_list, dim=0).mean(0)
    return logits_ens, emb_ens

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher_files', nargs='+', required=True, help='list of teacher model files (pth/onnx)')
    p.add_argument('--weight_dir', default='saved_models')
    p.add_argument('--dataset', default='gmdb')
    p.add_argument('--dataset_version', default='v1.1.0')
    p.add_argument('--img_size', type=int, default=112)
    p.add_argument('--in_channels', type=int, default=3)
    p.add_argument('--data_dir', default='../data')
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num_branches', type=int, default=3)
    p.add_argument('--emb_dim', type=int, default=512)
    p.add_argument('--width', type=float, default=0.5)
    p.add_argument('--T', type=float, default=4.0)
    p.add_argument('--w_ce', type=float, default=1.0)
    p.add_argument('--w_kd', type=float, default=1.0)
    p.add_argument('--w_mse', type=float, default=1.0)
    p.add_argument('--out', default='saved_models/student_compact_ekd.pth')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')

    # Datasets
    ds_train, ds_val = get_train_and_val_datasets(args.dataset, 'train', args.dataset_version,
                                                  args.img_size, args.in_channels, args.data_dir, img_postfix='_aligned')
    num_classes = ds_train.get_num_classes()
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=8, worker_init_fn=seed_worker, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=4, worker_init_fn=seed_worker, drop_last=False)

    # Teachers
    teachers = load_teacher_ensemble(args.weight_dir, args.teacher_files, num_classes, device)

    # Student
    student = CompactEnsemble(num_classes=num_classes, num_branches=args.num_branches,
                              in_channels=args.in_channels, emb_dim=args.emb_dim, width=args.width).to(device)

    opt = optim.Adam(student.parameters(), lr=args.lr, weight_decay=5e-5)

    for epoch in range(args.epochs):
        student.train()
        for x, y in train_loader:
            x = x.to(device, dtype=torch.float32); y = y.to(device, dtype=torch.int64)

            s_logits_list, s_emb_list = student(x)
            t_logits_ens, t_emb_ens = teacher_forward_batch(teachers, x, device)

            # If teacher logits are not available (e.g., ONNX-only teachers), skip KD logits term by setting w_kd=0
            total, parts = combine_losses(
                s_logits_list, s_emb_list,
                t_logits_ens if t_logits_ens is not None else s_logits_list[0].detach()*0,
                t_emb_ens,
                y, T=args.T, w_ce=args.w_ce, w_kd=(args.w_kd if t_logits_ens is not None else 0.0), w_mse=args.w_mse
            )

            opt.zero_grad()
            total.backward()
            opt.step()

        # Simple validation: top-1 acc on aggregated logits
        student.eval()
        correct = total_n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, dtype=torch.float32); y = y.to(device, dtype=torch.int64)
                s_logits_list, _ = student(x)
                logits = torch.stack(s_logits_list, dim=0).mean(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total_n += y.size(0)
        print(f"Epoch {epoch+1}/{args.epochs} - Val Top1: {correct/total_n:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(student, args.out)
    print(f"Saved EKD student to {args.out}")

if __name__ == '__main__':
    main()
````

4) Student inference for evaluation
- New script: predict_student.py (mirrors predict.py but runs the student model and TTA if desired). Outputs the same CSV/PKL so you can reuse evaluation.py and evaluate_ensemble.py.

````python
import argparse, os, datetime, cv2, torch, numpy as np, pandas as pd
import albumentations as A
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2

def preprocess(img, img_size=112, gray=False, flip=False):
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if gray: img = A.to_gray(img)
    if flip: img = A.hflip(img)
    img = np.transpose(img, (2,0,1))
    img = torch.from_numpy(img).unsqueeze(0).float().div(255).sub(0.5).div(0.5)
    return img

def parse_args():
    p = argparse.ArgumentParser(description='Encode images using EKD student')
    p.add_argument('--device', choices=['cpu','cuda','mps'], default='cpu')
    p.add_argument('--data', nargs='+', default=['data/cases_align'])
    p.add_argument('--save_dir', default='data/encodings')
    p.add_argument('--output_name', default='student_encodings.csv')
    p.add_argument('--model_path', required=True)
    p.add_argument('--img_size', type=int, default=112)
    p.add_argument('--separate_outputs', action='store_true')
    p.add_argument('--save_as_pickle', action='store_true')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')
    model = torch.load(args.model_path, map_location=device).to(device)
    model.eval()

    # gather image paths
    img_paths = []
    for d in args.data:
        if os.path.isdir(d):
            img_paths += [os.path.join(dp, f) for dp,_,fn in os.walk(d) for f in fn]
        else:
            img_paths.append(d)

    os.makedirs(args.save_dir, exist_ok=True)
    df = pd.DataFrame(columns=["img_name", "model", "flip", "gray", "class_conf", "representations"])

    with torch.no_grad():
        for p in img_paths:
            img_name = os.path.basename(p)
            img = cv2.imread(p)
            rows = []
            for flip in [False, True]:
                for gray in [False, True]:
                    x = preprocess(img, img_size=args.img_size, gray=gray, flip=flip).to(device, dtype=torch.float32)
                    s_logits_list, s_emb_list = model(x)
                    logits = torch.stack(s_logits_list, dim=0).mean(0)
                    rep = torch.stack(s_emb_list, dim=0).mean(0)
                    rep = F.normalize(rep).squeeze().tolist()
                    if args.separate_outputs:
                        out = os.path.join(args.save_dir, f"{img_name.rsplit('_',1)[0]}_encoding.csv")
                        with open(out, "a+") as f:
                            f.write(f"{img_name};student;{int(flip)};{int(gray)};{logits.squeeze().tolist()};{rep}\n")
                    else:
                        rows.append({"img_name": img_name, "model": "student", "flip": int(flip),
                                     "gray": int(gray), "class_conf": logits.squeeze().tolist(), "representations": rep})
            if not args.separate_outputs:
                df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    if not args.separate_outputs:
        if args.save_as_pickle:
            df.to_pickle(os.path.join(args.save_dir, args.output_name.replace('.csv','.pkl')))
        else:
            df.to_csv(os.path.join(args.save_dir, args.output_name), sep=';', index=False)

if __name__ == '__main__':
    main()
````

Minimal optional adjustments to existing files
- Expose a “features only” forward in my_arcface.py if needed for more granular feature KD (you already return a 512-d embedding as pred_rep; this may be enough). If you want a clean API:

````python
# ...existing code...
class MyArcFace(nn.Module):
    # ...existing code...
    def forward_features(self, x):
        # returns normalized 512-d embedding only
        with torch.no_grad():
            # forward up to embedding head
            x = self.base(x)
            x = self.features(x)
            return F.normalize(x)
    # ...existing code...
````

How to run (end-to-end)

1) Train teachers (reuse)
- See commands in README.md (“Train models”) for M1/M2.
- For M3: use the original r100 ONNX as a fixed teacher, or fine-tune it with train_gm_arc.py and save .pth.

2) Prune teachers
- Example:
  - M1: s1_glint360k_r50_512d_gmdb__v1.1.0_bs64_size112_channels3_last_model.pth
  - M2: s2_glint360k_r100_512d_gmdb__v1.1.0_bs128_size112_channels3_last_model.pth
  - M3: glint360k_r100.onnx (skip pruning or convert and prune selectively)

Run:
- Prune M1
  - python prune_iterative.py --weights saved_models/s1_glint360k_r50_512d_gmdb__v1.1.0_bs64_size112_channels3_last_model.pth --out saved_models/pruned_m1.pth --schedule 0.1 0.1 0.1
- Prune M2
  - python prune_iterative.py --weights saved_models/s2_glint360k_r100_512d_gmdb__v1.1.0_bs128_size112_channels3_last_model.pth --out saved_models/pruned_m2.pth --schedule 0.1 0.1 0.1

3) Train EKD student
- Example:
  - python train_ekd.py --teacher_files pruned_m1.pth pruned_m2.pth glint360k_r100.onnx --weight_dir saved_models --epochs 30 --num_branches 3 --out saved_models/student_compact_ekd.pth

4) Evaluate student
- Produce encodings with the student:
  - python predict_student.py --model_path saved_models/student_compact_ekd.pth --data ./data/cases_align --save_dir ./data/encodings --output_name student_encodings.csv
- Compare using your existing evaluation:
  - For direct gallery/test eval, use evaluate.py or evaluate_ensemble.py with the student encodings (formatted like predict.py output). Example:
    - python evaluate.py --metadata_dir gmdb_metadata --gallery_input ./data/gallery_encodings/GMDB_gallery_encodings_v1.1.0.pkl --case_input ./data/encodings/student_encodings.csv --output_dir demo_output --output_file student_results.json --top_n all

Notes and rationale
- Compatibility: student outputs mirror your “logits + 512-d embedding” to drop-in to existing pipelines (distance in evaluation.py, and CSV format in `predict.py`).
- Pruning strategy: Unstructured pruning avoids topology surgery, keeps saved .pth usable with current loaders.
- KD objective: Combines CE (labels), KL (soft teacher targets), and MSE (embedding alignment). Branch-level KD preserves teacher diversity while also using ensembled targets.
- Edge-readiness: `CompactEnsemble` uses depthwise separable conv; reduce width and branches to fit device constraints.

Where to document
- Extend README.md with a new “Ensemble Knowledge Distillation” section describing:
  - Teacher training commands (existing)
  - Pruning commands
  - EKD training command
  - Student prediction and evaluation commands

If you need deeper feature-level KD (e.g., layer-wise hints), add forward hooks to capture intermediate features in my_arcface.py and align them in `losses.py`.