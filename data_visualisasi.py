import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATASET_DIR = 'data/BTXRD'
file_path = os.path.join(DATASET_DIR, 'dataset.xlsx')  
df = pd.read_excel(file_path)

def determine_class(row):
    if row['tumor'] == 0:
        return 'normal'
    elif row['benign'] == 1:
        return 'benign'
    elif row['malignant'] == 1:
        return 'malignant'
    else:
        return 'unknown'

# Apply to DataFrame
df['class_label'] = df.apply(determine_class, axis=1)

# Count samples per class
class_counts = df['class_label'].value_counts()

# Display result
print("Class distribution:")
print(class_counts)

class_counts.plot(kind='bar', color='skyblue')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()