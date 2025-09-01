import argparse, json, os, random, math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from matplotlib.path import Path as MplPath
import pandas as pd

# ---- your project ops ----
from util import CLAHE
from utils_xray import WaveletDenoise, UnsharpMask

# =================== helpers ===================

def to_gray_f32(img_pil):
    # use luminance; returns HxW float32 in [0,1]
    if img_pil.mode != "L":
        img_pil = img_pil.convert("L")
    arr = np.asarray(img_pil, dtype=np.float32) / 255.0
    return arr

def resize_pil(img_pil, size):
    return transforms.Resize((size, size))(img_pil)

def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

def polygon_mask(h, w, poly_xy):
    # poly_xy: list[(x,y)] in display coords
    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.vstack((xx.ravel(), yy.ravel())).T
    path = MplPath(np.asarray(poly_xy, dtype=np.float32))
    mask = path.contains_points(coords).reshape(h, w)
    return mask

def rect_mask(h, w, x0, y0, x1, y1):
    xL, xR = int(round(min(x0,x1))), int(round(max(x0,x1)))
    yT, yB = int(round(min(y0,y1))), int(round(max(y0,y1)))
    m = np.zeros((h,w), dtype=bool)
    m[yT:yB+1, xL:xR+1] = True
    return m

def roi_mask_from_ann(ann, out_size):
    """Return ROI mask (bool HxW) scaled to out_size x out_size.
       Prefer polygon; fallback to first rectangle."""
    W0 = float(ann.get("imageWidth"))
    H0 = float(ann.get("imageHeight"))
    sx, sy = out_size / W0, out_size / H0
    H = W = out_size

    poly = None
    rect = None
    for s in ann.get("shapes", []):
        st = (s.get("shape_type") or "").lower()
        pts = s.get("points", [])
        if st == "polygon" and len(pts) >= 3 and poly is None:
            poly = [(x*sx, y*sy) for (x,y) in pts]
        if st == "rectangle" and len(pts) == 2 and rect is None:
            (x0,y0),(x1,y1) = pts
            rect = (x0*sx, y0*sy, x1*sx, y1*sy)

    if poly is not None:
        return polygon_mask(H, W, poly)
    if rect is not None:
        x0,y0,x1,y1 = rect
        return rect_mask(H, W, x0,y0,x1,y1)

    # fallback: entire image (shouldn't happen)
    return np.ones((H,W), dtype=bool)

def background_ring_from_mask(mask, ring=25):
    """Create a thin background ring around ROI, excluding ROI."""
    # bounding box of ROI
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    H, W = mask.shape
    y0r, y1r = max(0, y0-ring), min(H-1, y1+ring)
    x0r, x1r = max(0, x0-ring), min(W-1, x1+ring)

    ring_mask = np.zeros_like(mask)
    ring_mask[y0r:y1r+1, x0r:x1r+1] = True
    ring_mask[ y0:y1+1,  x0:x1+1] = False  # subtract ROI
    return ring_mask

def grad_mag(img):  # simple central diff, reflect edges
    gx = np.zeros_like(img, dtype=np.float32)
    gy = np.zeros_like(img, dtype=np.float32)
    gx[:,1:-1] = (img[:,2:] - img[:,:-2]) * 0.5
    gx[:,0]    = img[:,1] - img[:,0]
    gx[:,-1]   = img[:,-1] - img[:,-2]
    gy[1:-1,:] = (img[2:,:] - img[:-2,:]) * 0.5
    gy[0,:]    = img[1,:] - img[0,:]
    gy[-1,:]   = img[-1,:] - img[-2,:]
    return np.sqrt(gx*gx + gy*gy)

def conv2_laplacian(img):
    k = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
    pad = 1
    p = np.pad(img, ((pad,pad),(pad,pad)), mode="reflect")
    H,W = img.shape
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(H):
        for j in range(W):
            out[i,j] = np.sum(p[i:i+3, j:j+3] * k)
    return out

def entropy_bits(img_vals):
    # img_vals in [0,1] float
    hist, _ = np.histogram(img_vals, bins=256, range=(0,1), density=True)
    p = hist + 1e-12
    p = p / p.sum()
    return float(-(p*np.log2(p)).sum())

def compute_metrics(gray, roi_mask, bg_mask):
    eps = 1e-8
    roi = gray[roi_mask]
    bg  = gray[bg_mask]
    mu_r, sd_r = float(roi.mean()), float(roi.std(ddof=0))
    mu_b, sd_b = float(bg.mean()),  float(bg.std(ddof=0))

    cnr = (mu_r - mu_b) / (sd_b + eps)
    snr_roi = mu_r / (sd_r + eps)
    ent = entropy_bits(roi)

    g = grad_mag(gray)
    grad_mean = float(g[roi_mask].mean())

    lap = conv2_laplacian(gray)
    lap_var = float(lap[roi_mask].var())

    return {
        "Mean(ROI)": mu_r,
        "Std(ROI)": sd_r,
        "Mean(BG)": mu_b,
        "Std(BG)": sd_b,
        "CNR": cnr,
        "SNR(ROI)": snr_roi,
        "Entropy(ROI)": ent,
        "GradMean(ROI)": grad_mean,
        "LapVar(ROI)": lap_var,
    }

# =================== pipelines ===================

def pipeline_imgs(img_pil, size, clip=2.0, grid=(8,8), wavelet="db2", level=2,
                  unsharp_amount=0.7, unsharp_radius=1.0, unsharp_thr=2):
    resize = transforms.Resize((size, size))
    # Original
    p0 = resize(img_pil)

    # CLAHE
    p1 = resize(CLAHE(clip_limit=clip, tile_grid_size=grid, p=1.0)(img_pil))

    # CLAHE -> Wavelet
    p2 = resize(WaveletDenoise(wavelet=wavelet, level=level, p=1.0)(p1))

    # CLAHE -> Wavelet -> Unsharp
    p3 = resize(UnsharpMask(amount=unsharp_amount, radius=unsharp_radius, threshold=unsharp_thr, p=1.0)(p2))

    return [p0, p1, p2, p3], ["Original", "CLAHE", "CLAHE+Wavelet", "CLAHE+Wavelet+Unsharp"]

# =================== main routines ===================

def analyze_one(ann_path, images_root, size=600, params=None):
    ann = load_json(ann_path)
    img_path = os.path.join(images_root, ann["imagePath"])
    img = Image.open(img_path).convert("RGB")

    imgs, names = pipeline_imgs(
        img, size,
        clip=params["clip"], grid=params["grid"],
        wavelet=params["wavelet"], level=params["level"],
        unsharp_amount=params["unsharp_amount"],
        unsharp_radius=params["unsharp_radius"],
        unsharp_thr=params["unsharp_thr"],
    )
    gray_list = [to_gray_f32(p) for p in imgs]

    roi = roi_mask_from_ann(ann, size)
    bg  = background_ring_from_mask(roi, ring=25)

    rows = []
    for name, gray in zip(names, gray_list):
        m = compute_metrics(gray, roi, bg)
        m["Variant"] = name
        m["Image"] = ann.get("imagePath")
        rows.append(m)
    return pd.DataFrame(rows), imgs, names, roi

def barplot_from_df(df, title_suffix=""):
    # show key metrics relative to Original
    base = df[df["Variant"]=="Original"].iloc[0]
    metrics = ["CNR", "GradMean(ROI)", "LapVar(ROI)"]
    xs = np.arange(len(metrics))
    variants = df["Variant"].unique().tolist()

    plt.figure(figsize=(8,4))
    for i, var in enumerate(variants):
        row = df[df["Variant"]==var].iloc[0]
        vals = [row[m] for m in metrics]
        plt.plot(xs, vals, marker="o", label=var)
    plt.xticks(xs, metrics, rotation=0)
    plt.title(f"ROI metrics across variants{title_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.show()

def grid_show(imgs, names, roi_mask):
    n = len(imgs)
    plt.figure(figsize=(4.5*n, 4.5))
    for i,(im,name) in enumerate(zip(imgs, names), 1):
        plt.subplot(1,n,i)
        plt.imshow(im)
        yy, xx = np.where(roi_mask)
        if yy.size>0:
            y0,y1,x0,x1 = yy.min(), yy.max(), xx.min(), xx.max()
            plt.gca().add_patch(plt.Rectangle((x0,y0), x1-x0, y1-y0, fill=False, linewidth=2))
        plt.title(name); plt.axis("off")
    plt.tight_layout(); plt.show()

def find_ann_by_id(ann_dir, image_id):
    cand = os.path.join(ann_dir, f"{image_id}.json")
    if os.path.exists(cand): return cand
    for f in os.listdir(ann_dir):
        if f.lower().endswith(".json") and image_id.lower() in f.lower():
            return os.path.join(ann_dir, f)
    raise FileNotFoundError(f"Annotation for '{image_id}' not found in {ann_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_dir", required=True)
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--image_id", default=None, help="e.g., IMG0000203 (analyze exactly one)")
    ap.add_argument("--n", type=int, default=0, help="Random N (<=4) if image_id not given")
    ap.add_argument("--img_size", type=int, default=600)
    # ops params
    ap.add_argument("--clahe_clip", type=float, default=2.0)
    ap.add_argument("--clahe_grid", type=int, nargs=2, default=(8,8))
    ap.add_argument("--wavelet", type=str, default="db2")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--unsharp_amount", type=float, default=0.7)
    ap.add_argument("--unsharp_radius", type=float, default=1.0)
    ap.add_argument("--unsharp_thr", type=int, default=2)
    ap.add_argument("--save_csv", type=str, default=None)
    args = ap.parse_args()

    params = dict(
        clip=args.clahe_clip, grid=tuple(args.clahe_grid),
        wavelet=args.wavelet, level=args.level,
        unsharp_amount=args.unsharp_amount,
        unsharp_radius=args.unsharp_radius,
        unsharp_thr=args.unsharp_thr
    )

    dfs = []
    if args.image_id:
        ann_path = find_ann_by_id(args.ann_dir, args.image_id)
        df, imgs, names, roi = analyze_one(ann_path, args.images_root, args.img_size, params)
        dfs.append(df)
        # quick visuals
        grid_show(imgs, names, roi)
        barplot_from_df(df, title_suffix=f" — {args.image_id}")
        print("\nPer-variant metrics:")
        print(df[["Variant","CNR","SNR(ROI)","Entropy(ROI)","GradMean(ROI)","LapVar(ROI)"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    else:
        ann_files = [os.path.join(args.ann_dir,f) for f in os.listdir(args.ann_dir) if f.endswith(".json")]
        random.shuffle(ann_files)
        ann_files = ann_files[:max(1, min(4, args.n))]
        for apath in ann_files:
            df, _, _, _ = analyze_one(apath, args.images_root, args.img_size, params)
            dfs.append(df)

        big = pd.concat(dfs, ignore_index=True)
        # aggregate by Variant
        agg = big.groupby("Variant")[["CNR","SNR(ROI)","Entropy(ROI)","GradMean(ROI)","LapVar(ROI)"]].mean().reset_index()
        print("\nAggregated (mean over images):")
        print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        barplot_from_df(agg, title_suffix=" — aggregated")

    if args.save_csv:
        out = pd.concat(dfs, ignore_index=True)
        out.to_csv(args.save_csv, index=False)
        print(f"\nSaved detailed metrics to {args.save_csv}")

if __name__ == "__main__":
    main()
