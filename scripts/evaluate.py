import json, joblib
from sklearn.metrics import classification_report
from src.data import load_dataset
import yaml

cfg = yaml.safe_load(open("config/train.yaml"))
X, y = load_dataset(cfg)

model = joblib.load("artifacts/model.joblib")
pred = model.predict(X)

report = classification_report(y, pred, output_dict=True)
json.dump(report, open("artifacts/report.json", "w"), indent=2)

print("Evaluation OK")
