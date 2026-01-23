## Cat vs Dog Classifier

ResNet18 transfer-learning pipeline for the Kaggle Dogs vs Cats dataset. The script freezes the pretrained backbone, trains a 2-class head, and saves the best checkpoint based on validation accuracy.

### Requirements

```bash
pip install -r requirements.txt
```

### Data Preparation

The project expects an 80/10/10 split under `data/`:

```
data/
	train/
		cats/
		dogs/
	val/
		cats/
		dogs/
	test/
		cats/
		dogs/
```

You can create this via:

```bash
# If you have the Dogs vs Cats zip locally (e.g., kagglecatsanddogs_5340.zip renamed to dogs-vs-cats.zip)
python setup_data.py
```

On Kaggle, you can also organize from an attached dataset by copying cat/dog images into `organized_data/` and running `splitfolders` to build `data/` (see notebook snippets used previously).

### Training

```bash
python train.py
```

- Saves the best checkpoint to `best_model.pth`
- Logs train/val metrics per epoch
- Produces `training_curves.png` and `confusion_matrix.png`

### Evaluation

```bash
python evaluate.py
```

Reports test accuracy, classification report, and confusion matrix; saves `confusion_matrix.png`.

### Notes

- Large datasets and model weights are `.gitignore`d to keep the repo small.
- Uses ImageNet normalization and light augmentation (resize→crop, flip, rotation, color jitter) on training only.
