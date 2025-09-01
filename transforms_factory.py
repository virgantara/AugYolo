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
    use_wavelet = getattr(args, 'use_wavelet', False)
    use_unsharp = getattr(args, 'use_unsharp', False)
    use_clahe_aug = getattr(args, 'use_clahe', True)  # for B/G

    if args.scenario == 'G':
        # G = CLAHE as augmentation + (Wavelet + Unsharp) as preprocessing
        use_wavelet = True
        use_unsharp = True

    def make_preprocess(p_wavelet: float, p_unsharp: float, include_clahe_c: bool):
        ops = []
        if include_clahe_c:  # scenario C: CLAHE as preprocessing
            ops.append(CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0))
        ops.append(_maybe(
            WaveletDenoise(
                wavelet=getattr(args, 'wavelet_name', 'db2'),
                level=getattr(args, 'wavelet_level', 2),
                p=p_wavelet),
            use_wavelet
        ))
        ops.append(_maybe(
            UnsharpMask(
                amount=getattr(args, 'unsharp_amount', 0.7),
                radius=getattr(args, 'unsharp_radius', 1.0),
                threshold=getattr(args, 'unsharp_threshold', 2),
                p=p_unsharp),
            use_unsharp
        ))
        ops.append(transforms.Resize(img_size))
        return transforms.Compose(ops)

    is_C = (args.scenario == 'C')
    # train: keep user-provided probabilities
    preprocess_train = make_preprocess(
        p_wavelet=getattr(args, 'wavelet_p', 1.0),
        p_unsharp=getattr(args, 'unsharp_p', 1.0),
        include_clahe_c=is_C
    )
    # test: FORCE deterministic preprocessing
    preprocess_test = make_preprocess(
        p_wavelet=1.0,
        p_unsharp=1.0,
        include_clahe_c=is_C
    )

    # ================= AUG (train only) =================
    # aug_list = [
    #     transforms.RandomHorizontalFlip(0.5),
    #     transforms.RandomRotation(20),
    # ]
    aug_list = [
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),  # random crop + resize
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),      # brightness/contrast
    ]
    
    if args.scenario in ['B', 'G'] and use_clahe_aug:
        aug_list.insert(0, CLAHE(
            clip_limit=2.0, tile_grid_size=(8, 8),
            p=getattr(args, 'clahe_p', 0.25)
        ))
    if getattr(args, 'use_structuremap', False):
        aug_list.append(StructureMap(block_size=31, C=5, morph_open=1, morph_close=1, alpha=0.3, p=0.35))
    augment = transforms.Compose(aug_list)

    # ================= FINAL =================
    train_transform = transforms.Compose([
        preprocess_train,
        augment,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        preprocess_test,                 # deterministic; no aug
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, test_transform
