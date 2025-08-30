# BTXRD

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

## Training Classification
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

## Eval
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