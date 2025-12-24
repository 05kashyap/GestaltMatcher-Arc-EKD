import argparse
import torch
import os
import sys
from lib.pruning.prune_utils import prune_iterative
from lib.distill.teacher_loader import load_teacher
# Add lib to path to import dataset utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'lib')))
from datasets.utils import get_train_and_val_datasets
from utils_functions import seed_worker
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

def parse_args():
    p = argparse.ArgumentParser(description="Iterative pruning and fine-tuning script")
    p.add_argument('--weights', required=True, help="Path to the model weights file to be pruned.")
    p.add_argument('--out', required=True, help="Path to save the pruned model.")
    p.add_argument('--dataset', default='gmdb', help="Dataset name for fine-tuning.")
    p.add_argument('--dataset_version', default='v1.1.0', help="Dataset version.")
    p.add_argument('--img_size', type=int, default=112, help="Image size.")
    p.add_argument('--in_channels', type=int, default=3, help="Input image channels.")
    p.add_argument('--data_dir', default='./data', help="Directory for the dataset.")
    p.add_argument('--device', default='cuda', help="Device to use ('cuda' or 'cpu').")
    p.add_argument('--epochs_ft', type=int, default=3, help="Number of epochs for fine-tuning after pruning.")
    p.add_argument('--batch_size', type=int, default=64, help="Batch size for fine-tuning.")
    p.add_argument('--lr', type=float, default=5e-5, help="Learning rate for fine-tuning.")
    p.add_argument('--schedule', type=float, nargs='+', default=[0.1, 0.1, 0.1], help="List of pruning amounts per iteration.")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if 'cuda' in args.device and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model (note: teacher loader returns a frozen model, so we unfreeze it for fine-tuning)
    model = load_teacher(args.weights, device)
    for p in model.parameters():
        p.requires_grad = True

    # Perform iterative pruning
    print(f"Applying pruning with schedule: {args.schedule}")
    prune_iterative(model, schedule=args.schedule)
    print("Pruning complete.")

    # Fine-tune the pruned model
    if args.epochs_ft > 0 and hasattr(model, 'forward'): # Check if it's a trainable model
        print(f"Starting fine-tuning for {args.epochs_ft} epochs...")
        train_ds, _ = get_train_and_val_datasets(args.dataset, 'train', args.dataset_version,
                                                 args.img_size, args.in_channels, args.data_dir, img_postfix='_aligned.jpg')
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker, drop_last=True)
        
        opt = optim.Adam(model.parameters(), lr=args.lr)
        model.train()
        for e in range(args.epochs_ft):
            total_loss = 0
            for i, (x, y) in enumerate(train_loader):
                x = x.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.int64)
                
                # MyArcFace models return (logits, embedding)
                pred, _ = model(x)
                loss = F.cross_entropy(pred, y)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
            print(f"Epoch {e+1}/{args.epochs_ft}, Avg Loss: {total_loss / len(train_loader):.4f}")
    
    # Save the pruned and fine-tuned model
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model, args.out)
    print(f"Saved pruned model to {args.out}")

if __name__ == '__main__':
    main()