import argparse, os, cv2, torch, numpy as np, pandas as pd
import albumentations as A
import torch.nn.functional as F
from tqdm import tqdm

def preprocess(img, img_size=112, gray=False, flip=False):
    """Preprocesses an image for model inference."""
    if img is None:
        return None
    img = cv2.resize(img, (img_size, img_size))
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # back to 3 channels
    if flip:
        img = cv2.flip(img, 1)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float().div(255).sub(0.5).div(0.5)
    return img

def parse_args():
    p = argparse.ArgumentParser(description='Encode images using a trained EKD student model.')
    p.add_argument('--model_path', required=True, help="Path to the trained student model (.pth).")
    p.add_argument('--data', nargs='+', required=True, help="Path to an image file or a directory of images.")
    p.add_argument('--save_dir', default='data/encodings', help="Directory to save the output encodings.")
    p.add_argument('--output_name', default='student_encodings.pkl', help="Output filename (.pkl or .csv).")
    p.add_argument('--img_size', type=int, default=112)
    p.add_argument('--device', default='cuda')
    p.add_argument('--save_as_pickle', action='store_true', help="Save output as a pickle file (default).")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if 'cuda' in args.device and torch.cuda.is_available() else 'cpu')
    
    model = torch.load(args.model_path, map_location=device).to(device)
    model.eval()

    img_paths = []
    for d in args.data:
        if os.path.isdir(d):
            img_paths.extend([os.path.join(dp, f) for dp, _, fn in os.walk(d) for f in fn if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        else:
            img_paths.append(d)

    os.makedirs(args.save_dir, exist_ok=True)
    all_rows = []

    with torch.no_grad():
        for p in tqdm(img_paths, desc="Encoding images"):
            img_name = os.path.basename(p)
            img = cv2.imread(p)
            if img is None:
                print(f"Warning: Could not read image {p}, skipping.")
                continue

            # TTA: flip and color/gray
            for flip in [False, True]:
                for gray in [False, True]:
                    x = preprocess(img, img_size=args.img_size, gray=gray, flip=flip).to(device)
                    
                    s_logits_list, s_emb_list = model(x)
                    
                    # Aggregate results from student branches
                    logits = torch.stack(s_logits_list, dim=0).mean(0)
                    rep = torch.stack(s_emb_list, dim=0).mean(0)
                    rep = F.normalize(rep).squeeze().cpu().tolist()
                    
                    all_rows.append({"img_name": img_name, "model": "student", "flip": int(flip),
                                     "gray": int(gray), "class_conf": logits.squeeze().cpu().tolist(), 
                                     "representations": rep})

    df = pd.DataFrame(all_rows)
    
    output_path = os.path.join(args.save_dir, args.output_name)
    if args.save_as_pickle or output_path.endswith('.pkl'):
        df.to_pickle(output_path)
        print(f"Saved {len(df)} encodings to {output_path}")
    else:
        df.to_csv(output_path, sep=';', index=False)
        print(f"Saved {len(df)} encodings to {output_path}")

if __name__ == '__main__':
    main()