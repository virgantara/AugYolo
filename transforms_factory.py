# transforms_factory.py
from torchvision import transforms

# import your custom ops
from utils_xray import WaveletDenoise, UnsharpMask, StructureMap
from util import CLAHE  # if you use CLAHE in scenario B/C

def _maybe(op, use_it: bool):
    """Return identity if not used; otherwise the transform."""
    return op if use_it else transforms.Lambda(lambda x: x)

def build_transforms(args):
    """
    Scenarios:
      A: baseline (no CLAHE)
      B: CLAHE as augmentation (optional, via args.use_clahe)
      C: CLAHE as preprocessing (always if scenario C)
      D: Research stack (Wavelet/Unsharp/StructureMap) – but now toggleable
      G: CLAHE as augmentation + Wavelet + Unsharp (preprocessing)

    Toggles (work in ANY scenario):
      --use_wavelet, --use_unsharp (apply as PREPROCESSING before resize)
      --wavelet_p, --unsharp_p: probabilities (1.0 => deterministic preprocessing)
      You can thus test each augmentation alone or in combination.

    For fairness, test_transform mirrors the preprocessing part only.
    """
    img_size = (args.img_size, args.img_size)

    # ======== PREPROCESSING (always before resize) ========
    pre_list = []

    # Scenario C has fixed CLAHE preprocessing; B can add CLAHE as aug later
    if args.scenario == 'C':
        pre_list.append(CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0))


    if args.scenario == 'G':
        args.use_wavelet = True
        args.use_unsharp = True
        
    # Optional wavelet & unsharp (can be used in ANY scenario)
    pre_list.append(_maybe(
        WaveletDenoise(wavelet=getattr(args, 'wavelet_name', 'db2'),
                       level=getattr(args, 'wavelet_level', 2),
                       p=getattr(args, 'wavelet_p', 1.0)),
        getattr(args, 'use_wavelet', False)
    ))

    pre_list.append(_maybe(
        UnsharpMask(amount=getattr(args, 'unsharp_amount', 0.7),
                    radius=getattr(args, 'unsharp_radius', 1.0),
                    threshold=getattr(args, 'unsharp_threshold', 2),
                    p=getattr(args, 'unsharp_p', 1.0)),
        getattr(args, 'use_unsharp', False)
    ))

    # Always resize after preprocessing ops
    pre_list.append(transforms.Resize(img_size))
    preprocess = transforms.Compose(pre_list)

    # ======== AUGMENTATION (training only) ========
    aug_list = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
    ]

    # Scenario B: optional CLAHE as augmentation (weak + rare recommended)
    if args.scenario in ['B', 'G'] and getattr(args, 'use_clahe', True):
        aug_list.insert(0, CLAHE(
            clip_limit=2.0, tile_grid_size=(8, 8),
            p=getattr(args, 'clahe_p', 0.25)
        ))

    # Optional structure map (kept off by default; turn on to test)
    if getattr(args, 'use_structuremap', False):
        aug_list.append(
            StructureMap(block_size=31, C=5, morph_open=1, morph_close=1, alpha=0.3, p=0.35)
        )

    augment = transforms.Compose(aug_list)

    # ======== FINAL PIPELINES ========
    train_transform = transforms.Compose([
        preprocess,
        augment,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        preprocess,             # ONLY preprocessing, no train-time aug
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, test_transform