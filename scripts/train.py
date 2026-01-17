import yaml, json, joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from src.data import load_dataset
from src.features import build_numeric_preprocess
from src.model import build_model

def load_cfg():
    return yaml.safe_load(open("config/train.yaml"))

def save_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    cfg = load_cfg()
    art_dir = Path(cfg["artifacts_dir"])
    art_dir.mkdir(exist_ok=True)

    X, y = load_dataset(cfg)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        stratify=y
    )

    pipe = Pipeline(steps=[
        ("preprocess", build_numeric_preprocess()),
        ("model", build_model(cfg))
    ])

    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    acc = float(accuracy_score(yte, pred))
    f1  = float(f1_score(yte, pred, average="macro"))

    joblib.dump(pipe, art_dir / "model.joblib")
    json.dump({"accuracy": acc, "f1_macro": f1},
              open(art_dir / "metrics.json", "w"), indent=2)

    save_confusion_matrix(yte, pred, art_dir / "confusion_matrix.png")

    print("Train OK:", {"accuracy": acc, "f1_macro": f1})

if __name__ == "__main__":
    main()
