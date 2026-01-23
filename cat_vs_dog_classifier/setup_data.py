import os
import zipfile
import shutil
import splitfolders

# 1. Download from Kaggle
print("Downloading dataset...")
os.system("kaggle competitions download -c dogs-vs-cats")

# 2. Extract the main zip
print("Extracting main zip...")
with zipfile.ZipFile('dogs-vs-cats.zip', 'r') as zip_ref:
    zip_ref.extractall('base_data')

# 3. Extract the train.zip (contains the actual images)
print("Extracting images...")
with zipfile.ZipFile('base_data/train.zip', 'r') as zip_ref:
    zip_ref.extractall('raw_images')

# 4. Organize into Class Folders
print("Organizing into cats and dogs folders...")
original_dir = 'raw_images/train'
organized_dir = 'organized_data'

os.makedirs(f'{organized_dir}/cats', exist_ok=True)
os.makedirs(f'{organized_dir}/dogs', exist_ok=True)

for filename in os.listdir(original_dir):
    if filename.startswith('cat'):
        shutil.move(os.path.join(original_dir, filename), f'{organized_dir}/cats/{filename}')
    elif filename.startswith('dog'):
        shutil.move(os.path.join(original_dir, filename), f'{organized_dir}/dogs/{filename}')

# 5. Split into Train/Val/Test (80/10/10)
print("Splitting folders...")
splitfolders.ratio(organized_dir, output="data", seed=42, ratio=(.8, .1, .1))

print("Done! Your data is in the 'data/' folder.")