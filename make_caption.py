import pandas as pd
import os
from pathlib import Path
import random

DATASET_DIR = 'data/BTXRD'
metadata_xlsx_path = os.path.join(DATASET_DIR, 'dataset.xlsx')
# train_path = os.path.join(DATASET_DIR, 'train.xlsx')
# test_path = os.path.join(DATASET_DIR, 'val.xlsx')  
# IMG_DIR = os.path.join(DATASET_DIR, 'images')
df = pd.read_excel(metadata_xlsx_path)

# Define groups of columns
ANATOMY = ["humerus","radius","ulna","femur","tibia","fibula","foot","hand","hip bone","pelvis"]
JOINTS = ["ankle-joint","knee-joint","hip-joint","wrist-joint","elbow-joint","shoulder-joint"]
VIEW = ["frontal","lateral","oblique"]
TUMOR = ["osteochondroma","multiple osteochondromas","simple bone cyst","giant cell tumor",
         "osteofibroma","synovial osteochondroma","osteosarcoma","other bt","other mt"]
META = ["upper limb","lower limb"]
BENIGN_VARIANTS = [
    "showing a benign bone tumor",
    "containing a benign bone tumor",
    "with evidence of a benign tumor",
    "radiograph of a benign tumor",
    "depicting a benign tumor",
    "demonstrating a benign lesion"
]

MALIGNANT_VARIANTS = [
    "showing a malignant bone tumor",
    "containing a malignant bone tumor",
    "with evidence of a malignant tumor",
    "radiograph of a malignant tumor",
    "depicting a malignant tumor",
    "demonstrating a malignant lesion"
]

NO_TUMOR_VARIANTS = [
    "with no visible tumor",
    "showing no signs of tumor",
    "demonstrating absence of tumor",
    "radiograph without tumor evidence",
    "depicting a healthy bone with no tumor",
    "containing no tumor"
]

# Function to build captions from each row
def make_caption(row):
    parts = []
    # view
    v = [v for v in VIEW if row.get(v,0)==1]
    if v: 
        parts.append(f"{v[0]} x-ray")
    else: 
        parts.append("x-ray")
    
    # anatomy / site
    site = [a for a in ANATOMY if row.get(a,0)==1] + [j for j in JOINTS if row.get(j,0)==1]
    if site:
        parts.append("of the " + ", ".join(site))
    
    # limb meta
    limb = [m for m in META if row.get(m,0)==1]
    if limb:
        parts.append(f"({', '.join(limb)})")
    
    # diagnosis
    if row.get("tumor",0)==1:
        if row.get("benign",0)==1: 
            parts.append(random.choice(BENIGN_VARIANTS))
        if row.get("malignant",0)==1: 
            parts.append(random.choice(MALIGNANT_VARIANTS))
        diag = [t for t in TUMOR if row.get(t,0)==1]
        if diag:
            parts.append("with " + ", ".join(diag))
    else:
        parts.append(random.choice(NO_TUMOR_VARIANTS))
    
    return " ".join(parts)

# Apply to the dataframe
df["caption"] = df.apply(make_caption, axis=1)

# Save new file
output_path = Path("dataset_with_captions.xlsx")
df.to_excel(output_path, index=False)

print(f"Saved to {output_path}")
