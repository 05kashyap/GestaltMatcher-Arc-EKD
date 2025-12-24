import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
from typing import List, Optional, Tuple

# Add lib to path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'lib')))
from models.compact_net import CompactEnsemble
from distill.teacher_loader import load_teacher_ensemble
from distill.losses import combine_losses
from datasets.utils import get_train_and_val_datasets
from utils_functions import seed_worker

@torch.no_grad()
def teacher_forward_batch(teachers: List[torch.nn.Module], x: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    logits_list, emb_list = [], []
    for t in teachers:
        out = t(x)
        if isinstance(out, (list, tuple)) and len(out) == 2:
            logits, emb = out
            logits_list.append(logits)
        else: # Assumes embedding-only output for ONNX models
            emb = out
        emb_list.append(emb if emb.ndim == 2 else emb.squeeze())
    
    logits_ens = torch.stack(logits_list, dim=0).mean(0) if logits_list else None
    emb_ens = torch.stack(emb_list, dim=0).mean(0)
    return logits_ens, emb_ens

def parse_args():
    p = argparse.ArgumentParser(description="Ensemble Knowledge Distillation Trainer")
    p.add_argument('--teacher_files', nargs='+', required=True, help='List of teacher model files (pth/onnx).')
    p.add_argument('--weight_dir', default='saved_models', help="Directory for teacher weights.")
    p.add_argument('--out', default='saved_models/student_compact_ekd.pth', help="Output path for the student model.")
    # Dataset args
    p.add_argument('--dataset', default='gmdb')
    p.add_argument('--dataset_version', default='v1.1.0')
    p.add_argument('--data_dir', default='./data')
    # Model args
    p.add_argument('--num_branches', type=int, default=3)
    p.add_argument('--emb_dim', type=int, default=512)
    p.add_argument('--width', type=float, default=0.5, help="Width multiplier for student model.")
    p.add_argument('--img_size', type=int, default=112)
    p.add_argument('--in_channels', type=int, default=3)
    # Training args
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    # Loss args
    p.add_argument('--T', type=float, default=4.0, help="Temperature for KL divergence.")
    p.add_argument('--w_ce', type=float, default=1.0, help="Weight for Cross-Entropy loss.")
    p.add_argument('--w_kd', type=float, default=1.0, help="Weight for KL divergence loss.")
    p.add_argument('--w_mse', type=float, default=1.0, help="Weight for MSE embedding loss.")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if 'cuda' in args.device and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ds_train, ds_val = get_train_and_val_datasets(args.dataset, 'train', args.dataset_version,
                                                  args.img_size, args.in_channels, args.data_dir, img_postfix='_aligned.jpg')
    num_classes = ds_train.get_num_classes()
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=2, worker_init_fn=seed_worker, drop_last=False)

    print("Loading teacher ensemble...")
    teachers = load_teacher_ensemble(args.weight_dir, args.teacher_files, device)
    print(f"Loaded {len(teachers)} teachers.")

    student = CompactEnsemble(num_classes=num_classes, num_branches=args.num_branches,
                              in_channels=args.in_channels, emb_dim=args.emb_dim, width=args.width).to(device)
    opt = optim.Adam(student.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        student.train()
        for x, y in train_loader:
            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.int64)

            s_logits_list, s_emb_list = student(x)
            t_logits_ens, t_emb_ens = teacher_forward_batch(teachers, x)

            w_kd = args.w_kd if t_logits_ens is not None else 0.0
            total, parts = combine_losses(s_logits_list, s_emb_list, t_logits_ens, t_emb_ens, y,
                                          T=args.T, w_ce=args.w_ce, w_kd=w_kd, w_mse=args.w_mse)
            opt.zero_grad()
            total.backward()
            opt.step()

        student.eval()
        correct = total_n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.int64)
                s_logits_list, _ = student(x)
                logits = torch.stack(s_logits_list, dim=0).mean(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total_n += y.size(0)
        print(f"Epoch {epoch+1}/{args.epochs} - Val Top1 Acc: {correct/total_n:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(student, args.out)
    print(f"Saved EKD student to {args.out}")

if __name__ == '__main__':
    main()