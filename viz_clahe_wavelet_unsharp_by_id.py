import argparse, json, os, glob
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from PIL import Image
from torchvision import transforms

# project ops (same imports you use in training)
from util import CLAHE
from utils_xray import WaveletDenoise, UnsharpMask

def to_rgb(img):
    return img.convert("RGB") if img.mode != "RGB" else img

def draw_shapes(ax, shapes, sx, sy, lw=2.0):
    for s in shapes:
        pts = s.get("points", [])
        stype = (s.get("shape_type") or "").lower()
        if stype == "rectangle" and len(pts) == 2:
            (x1, y1), (x2, y2) = pts
            x1s, y1s = x1 * sx, y1 * sy
            x2s, y2s = x2 * sx, y2 * sy
            x0, y0 = min(x1s, x2s), min(y1s, y2s)
            w, h = abs(x2s - x1s), abs(y2s - y1s)
            ax.add_patch(Rectangle((x0, y0), w, h, fill=False, linewidth=lw))
            # if s.get("label"):
            #     ax.text(x0, max(0, y0 - 5), s["label"], fontsize=9,
            #             bbox=dict(fc="white", ec="none", alpha=0.7))
        elif stype in ("polygon", "polyline") and len(pts) >= 3:
            scaled = [(x * sx, y * sy) for x, y in pts]
            # ax.add_patch(Polygon(scaled, fill=False, linewidth=lw))
            # if s.get("label"):
            #     xm = sum(p[0] for p in scaled) / len(scaled)
            #     ym = sum(p[1] for p in scaled) / len(scaled)
            #     ax.text(xm, ym, s["label"], fontsize=9,
            #             bbox=dict(fc="white", ec="none", alpha=0.7))

def find_ann_path(ann_dir: str, image_id: str) -> str:
    cand = os.path.join(ann_dir, f"{image_id}.json")
    if os.path.exists(cand):
        return cand
    for p in glob.glob(os.path.join(ann_dir, "*.json")):
        base = os.path.splitext(os.path.basename(p))[0]
        if base.lower() == image_id.lower():
            return p
    subs = [p for p in glob.glob(os.path.join(ann_dir, "*.json"))
            if image_id.lower() in os.path.basename(p).lower()]
    if subs:
        return sorted(subs)[0]
    raise FileNotFoundError(f"No JSON found for ID '{image_id}' in {ann_dir}")

def resize_only(img, size):
    return transforms.Resize((size, size))(img)

def clahe(img, clip, grid):
    return CLAHE(clip_limit=clip, tile_grid_size=grid, p=1.0)(img)

def wavelet(img, name, level):
    return WaveletDenoise(wavelet=name, level=level, p=1.0)(img)

def unsharp(img, amount, radius, threshold):
    return UnsharpMask(amount=amount, radius=radius, threshold=threshold, p=1.0)(img)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_id", required=True, help="e.g., IMG0000203")
    ap.add_argument("--ann_dir", required=True, help="Folder with LabelMe JSONs")
    ap.add_argument("--images_root", required=True, help="Folder with images")
    ap.add_argument("--img_size", type=int, default=600)

    # ops params
    ap.add_argument("--clahe_clip", type=float, default=2.0)
    ap.add_argument("--clahe_grid", type=int, nargs=2, default=(8, 8))
    ap.add_argument("--wavelet_name", type=str, default="db2")
    ap.add_argument("--wavelet_level", type=int, default=2)
    ap.add_argument("--unsharp_amount", type=float, default=0.7)
    ap.add_argument("--unsharp_radius", type=float, default=1.0)
    ap.add_argument("--unsharp_threshold", type=int, default=2)

    # optional: change order (default = CLAHE first, as requested)
    ap.add_argument("--order", choices=["clahe-first", "wavelet-first"],
                    default="clahe-first",
                    help="Pipeline for combined panels. 'clahe-first' -> CLAHE→Wavelet→Unsharp. "
                         "'wavelet-first' -> Wavelet→CLAHE→Unsharp.")
    ap.add_argument("--save_path", type=str, default=None)
    args = ap.parse_args()

    # --- load annotation ---
    ann_path = find_ann_path(args.ann_dir, args.image_id)
    with open(ann_path, "r") as f:
        ann = json.load(f)

    # --- resolve image path from JSON (preferred), else try common suffixes ---
    image_rel = ann.get("imagePath")
    if not image_rel:
        for ext in (".jpeg", ".jpg", ".png"):
            test = os.path.join(args.images_root, args.image_id + ext)
            if os.path.exists(test):
                image_rel = os.path.basename(test); break
    img_path = os.path.join(args.images_root, image_rel)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    # --- load image ---
    img_raw = to_rgb(Image.open(img_path))

    # --- build four panels ---
    resize = transforms.Resize((args.img_size, args.img_size))

    # 1) Original (resized)
    p1 = resize(img_raw)

    # 2) CLAHE only
    p2 = resize(clahe(img_raw, args.clahe_clip, tuple(args.clahe_grid)))

    # 3) CLAHE + Wavelet  (default order: CLAHE then Wavelet)
    if args.order == "clahe-first":
        img_cw = wavelet(clahe(img_raw, args.clahe_clip, tuple(args.clahe_grid)),
                         args.wavelet_name, args.wavelet_level)
    else:  # wavelet-first
        img_cw = clahe(wavelet(img_raw, args.wavelet_name, args.wavelet_level),
                       args.clahe_clip, tuple(args.clahe_grid))
    p3 = resize(img_cw)

    # 4) CLAHE + Wavelet + Unsharp
    if args.order == "clahe-first":
        img_cwu = unsharp(
            wavelet(clahe(img_raw, args.clahe_clip, tuple(args.clahe_grid)),
                    args.wavelet_name, args.wavelet_level),
            args.unsharp_amount, args.unsharp_radius, args.unsharp_threshold
        )
    else:
        img_cwu = unsharp(
            clahe(wavelet(img_raw, args.wavelet_name, args.wavelet_level),
                  args.clahe_clip, tuple(args.clahe_grid)),
            args.unsharp_amount, args.unsharp_radius, args.unsharp_threshold
        )
    p4 = resize(img_cwu)

    # --- scale for overlays ---
    w0 = float(ann.get("imageWidth", p1.width))
    h0 = float(ann.get("imageHeight", p1.height))
    sx, sy = args.img_size / w0, args.img_size / h0

    # --- plot ---
    panels = [p1, p2, p3, p4]
    titles  = [
        f"Original (Resized) — {os.path.basename(image_rel)}",
        "CLAHE",
        f"CLAHE + Wavelet ({args.order})",
        f"CLAHE + Wavelet + Unsharp ({args.order})"
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, im, ttl in zip(axes, panels, titles):
        ax.imshow(im); ax.set_title(ttl); ax.axis("off")
        draw_shapes(ax, ann.get("shapes", []), sx, sy)

    plt.tight_layout()
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        plt.savefig(args.save_path, dpi=200, bbox_inches="tight")
        print(f"Saved to {args.save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
