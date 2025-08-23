import pandas as pd
import os
from pathlib import Path
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

model_name = "google/flan-t5-base"  # small and effective
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

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

def build_prompt(row):
    parts = []
    v = [v for v in VIEW if row.get(v,0)==1]
    if v: parts.append(f"{v[0]} x-ray")
    else: parts.append("x-ray")

    site = [a for a in ANATOMY if row.get(a,0)==1] + [j for j in JOINTS if row.get(j,0)==1]
    if site: parts.append("of the " + ", ".join(site))

    limb = [m for m in META if row.get(m,0)==1]
    if limb: parts.append(f"({', '.join(limb)})")

    if row.get("tumor",0)==1:
        if row.get("benign",0)==1:
            parts.append("showing a benign tumor")
        if row.get("malignant",0)==1:
            parts.append("showing a malignant tumor")
        diag = [t for t in TUMOR if row.get(t,0)==1]
        if diag:
            parts.append("type: " + ", ".join(diag))
    else:
        parts.append("no tumor detected")
    
    return " ".join(parts)

prompts = df.apply(build_prompt, axis=1).tolist()

# Step 2: Generate captions using FLAN-T5
generated_captions = []
for prompt in tqdm(prompts, desc="Generating captions"):
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
    output_ids = model.generate(input_ids, max_length=40, num_beams=3, do_sample=True)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_captions.append(caption)

# Step 3: Save back to Excel
df["caption"] = generated_captions
df.to_excel("dataset_with_transformer_captions.xlsx", index=False)

print("Saved dataset_with_transformer_captions.xlsx")