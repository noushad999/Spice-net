import sys
sys.path.insert(0, "/mnt/d/SpiceNet")

import config
from src.dataset import build_splits
from src.baselines import train_svm, build_svm_features
from src.utils import compute_and_print_metrics, plot_confusion_matrix, save_metrics

x_tr, y_tr, x_val, y_val, x_te, y_te = build_splits(config.DATA_DIR)

val_acc, pipe = train_svm(x_tr, y_tr, x_val, y_val)

print("Extracting test features...")
X_te = build_svm_features(x_te)
y_pred = pipe.predict(X_te).tolist()

metrics = compute_and_print_metrics(y_te, y_pred, config.CLASSES)
metrics["model"] = "svm_hog_color"

plot_confusion_matrix(y_te, y_pred, config.CLASSES, config.OUTPUT_DIR, prefix="svm")
save_metrics(metrics, config.OUTPUT_DIR / "svm_metrics.json")
print("Done!")
