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