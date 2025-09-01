import argparse, json, os, random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from PIL import Image
from torchvision import transforms
from util import CLAHE  # your project CLAHE

def to_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img

def load_annotation(ann_path: str):
    with open(ann_path, "r") as f:
        return json.load(f)

def apply_clahe_and_resize(img, img_size, clahe_before_resize=True):
    clahe = CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)
    resize = transforms.Resize((img_size, img_size))
    if clahe_before_resize:
        after = resize(clahe(img))
        before = resize(img)
    else:
        before = resize(img)
        after = clahe(before)
    return before, after

def draw_shapes(ax, shapes, scale_x, scale_y, lw=2.0):
    for s in shapes:
        pts = s.get("points", [])
        stype = s.get("shape_type", "").lower()
        if stype == "rectangle" and len(pts) == 2:
            (x1, y1), (x2, y2) = pts
            x1s, y1s = x1 * scale_x, y1 * scale_y
            x2s, y2s = x2 * scale_x, y2 * scale_y
            w, h = abs(x2s - x1s), abs(y2s - y1s)
            x0, y0 = min(x1s, x2s), min(y1s, y2s)
            ax.add_patch(Rectangle((x0, y0), w, h, fill=False, linewidth=lw, edgecolor="red"))
        elif stype in ("polygon", "polyline") and len(pts) >= 3:
            scaled = [(x * scale_x, y * scale_y) for x, y in pts]
            # ax.add_patch(Polygon(scaled, fill=False, linewidth=lw, edgecolor="lime"))

def visualize_random(ann_dir, images_root, img_size=600, n=2, clahe_before_resize=True):
    ann_files = [os.path.join(ann_dir, f) for f in os.listdir(ann_dir) if f.endswith(".json")]
    chosen = random.sample(ann_files, min(n, len(ann_files)))

    fig, axes = plt.subplots(len(chosen), 2, figsize=(10, 5 * len(chosen)))

    if len(chosen) == 1:  # make axes iterable if only 1 row
        axes = [axes]

    for row, ann_path in zip(axes, chosen):
        ann = load_annotation(ann_path)
        image_rel = ann["imagePath"]
        img_path = os.path.join(images_root, image_rel)
        img = Image.open(img_path)
        img = to_rgb(img)

        before, after = apply_clahe_and_resize(img, img_size, clahe_before_resize)
        sx, sy = img_size / ann["imageWidth"], img_size / ann["imageHeight"]

        # before
        row[0].imshow(before)
        row[0].set_title(f"Before: {os.path.basename(image_rel)}")
        row[0].axis("off")
        draw_shapes(row[0], ann.get("shapes", []), sx, sy)

        # after
        row[1].imshow(after)
        row[1].set_title("After CLAHE")
        row[1].axis("off")
        draw_shapes(row[1], ann.get("shapes", []), sx, sy)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_dir", type=str, required=True, default='data/BTXRD/Annotations', help="Directory with JSON annotations")
    ap.add_argument("--images_root", type=str, required=True, default='data/BTXRD/images', help="Directory with images")
    ap.add_argument("--img_size", type=int, default=600)
    ap.add_argument("--n", type=int, default=2, help="Number of random samples (max 4)")
    ap.add_argument("--clahe_after_resize", action="store_true", help="Use Resize -> CLAHE order")
    args = ap.parse_args()

    visualize_random(
        ann_dir=args.ann_dir,
        images_root=args.images_root,
        img_size=args.img_size,
        n=min(args.n, 4),
        clahe_before_resize=not args.clahe_after_resize
    )
