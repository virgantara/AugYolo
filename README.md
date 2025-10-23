# AugYolo
This is the implementation of our paper entitled "Enhancing Bone Tumor X-ray Classification with Hybrid Augmentation based on YOLOv8"

## Instalasi
1. Download dataset dari [sini](https://figshare.com/articles/dataset/A_Radiograph_Dataset_for_the_Classification_Localization_and_Segmentation_of_Primary_Bone_Tumors/27865398)

1. Jalan perintah ini: 
```bash
pip install -r requirements.txt
```

1. Instal conda environtment
```bash
conda install -n btxrd-env python=3.9
```

1. Aktivasi conda environment
```bash
conda activate btxrd-env
```

# Training Classification

## Train T-Test (Center 1 and 2 as training, Center 3 as testing)
```bash
python main_center_cls.py \
  --exp_name=exp_yolov8_ep300_600x600_sceG_ttest \
  --img_size=600 \
  --epochs=300 \
  --model_name=yolov8 \
  --batch_size=16 \
  --test_batch_size=16 \
  --seeds 42 43 44 45 46 \
  --scenario=G \
  --center_id=1 \
  --use_clahe --clahe_p=0.25 \
  --use_wavelet --wavelet_name=db2 --wavelet_level=1 --wavelet_p=1.0 \
  --use_unsharp --unsharp_amount=0.5 --unsharp_radius=0.8 --unsharp_threshold=2 --unsharp_p=1.0
```

## Train Whole
1. Jalankan perintah berikut
```bash
python main_cls.py --epochs=300 --batch_size=32 --test_batch_size=32
```

1. Run model Yolov8
```bash
python main_cls.py --exp_name=exp_yolov8 --img_size=608 --epochs=300 --model_name=yolov8 --batch_size=16 --test_batch_size=16 --seed=42
```

1. Scenario A (Baseline)
```bash
python main_cls.py --exp_name=exp_yolov8_ep300_600x600_sceA --img_size=600 --epochs=300 --model_name=yolov8 --batch_size=16 --test_batch_size=16 --seed=42 --scenario=A
```
2. Scenario B (CLAHE-only as Augmentation)
```bash
python main_cls.py --exp_name=exp_yolov8_ep300_600x600_sceB --img_size=600 --epochs=300 --model_name=yolov8 --batch_size=16 --test_batch_size=16 --seed=42 --scenario=B
```
3. Scenario C (CLAHE-only as preprocessing)
```bash
python main_cls.py --exp_name=exp_yolov8_ep300_600x600_sceC --img_size=600 --epochs=300 --model_name=yolov8 --batch_size=16 --test_batch_size=16 --seed=42 --scenario=C
```
4. Scenario D (Wavelet-only)
```bash
python main_cls.py --exp_name=exp_yolov8_ep300_600x600_sceD --img_size=600 --epochs=300 --model_name=yolov8 --batch_size=16 --test_batch_size=16 --seed=42 --scenario=D --use_wavelet --wavelet_p 1.0 
```

5. Scenario E (Unsharp-only)
```bash
python main_cls.py --exp_name=exp_yolov8_ep300_600x600_sceE --img_size=600 --epochs=300 --model_name=yolov8 --batch_size=16 --test_batch_size=16 --seed=42 --scenario=E --use_unsharp --unsharp_p 1.0 
```
6. Scenario F (Wavelet + Unsharp)
```bash
python main_cls.py --exp_name=exp_yolov8_ep300_600x600_sceF --img_size=600 --epochs=300 --model_name=yolov8 --batch_size=16 --test_batch_size=16 --seed=42 --scenario=F --use_wavelet --wavelet_p 1.0 --use_unsharp --unsharp_p 1.0
```
7. Scenario G (CLAHE + Wavelet + Unsharp)
```bash
python main_cls.py \
  --exp_name=exp_yolov8_ep300_600x600_sceG \
  --img_size=600 \
  --epochs=300 \
  --model_name=yolo \
  --batch_size=16 \
  --test_batch_size=16 \
  --seed=42 \
  --scenario=G \
  --use_clahe --clahe_p=0.25 \
  --use_wavelet --wavelet_name=db2 --wavelet_level=1 --wavelet_p=1.0 \
  --use_unsharp --unsharp_amount=0.5 --unsharp_radius=0.8 --unsharp_threshold=2 --unsharp_p=1.0

```

## Eval

### Best Scenario
```bash
bash run_eval.sh 
```
4. Scenario D (Wavelet-only)
```bash
python main_cls.py --exp_name=exp_yolov8_ep300_600x600_sceD --img_size=600 --epochs=300 --model_name=yolov8 --batch_size=16 --test_batch_size=16 --seed=42 --scenario=D --use_wavelet --wavelet_p 1.0 
```

5. Scenario E (Unsharp-only)
```bash
python eval.py --exp_name=exp_yolov8_ep300_600x600_sceE --img_size=600  --model_name=yolov8  --test_batch_size=16 --seed=42 --scenario=E --use_unsharp --unsharp_p 1.0 --model_path=checkpoints/exp_yolov8_ep300_600x600_sceE/best_model.pth
```
6. Scenario F (Wavelet + Unsharp)
```bash
python main_cls.py --exp_name=exp_yolov8_ep300_600x600_sceF --img_size=600 --epochs=300 --model_name=yolov8 --batch_size=16 --test_batch_size=16 --seed=42 --scenario=F --use_wavelet --wavelet_p 1.0 --use_unsharp --unsharp_p 1.0
```

