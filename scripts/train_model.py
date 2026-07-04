
import os, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
from huggingface_hub import HfApi, login

HF_TOKEN = os.environ["HF_TOKEN"]
HF_USERNAME = "harshverma27"
DATASET_REPO = f"{HF_USERNAME}/engine-sensor-data"
MODEL_REPO = f"{HF_USERNAME}/predictive-maintenance-engine"

login(token=HF_TOKEN, add_to_git_credential=False)
api = HfApi()

FEATURE_COLS = ["Engine rpm","Lub oil pressure","Fuel pressure",
                "Coolant pressure","lub oil temp","Coolant temp"]
TARGET_COL = "Engine Condition"

train_df = pd.read_csv(f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/data/train.csv")
test_df = pd.read_csv(f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/data/test.csv")
X_tr, y_tr = train_df[FEATURE_COLS], train_df[TARGET_COL]
X_te, y_te = test_df[FEATURE_COLS], test_df[TARGET_COL]

param_grid = {"n_estimators":[100,200], "max_depth":[5,10,None], "min_samples_split":[2,5]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gs = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), param_grid, cv=cv, scoring="f1", n_jobs=-1)
gs.fit(X_tr, y_tr)

best_model = gs.best_estimator_
f1 = f1_score(y_te, best_model.predict(X_te))
print(f"Best params: {gs.best_params_} | Test F1: {f1:.4f}")

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")

api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj="models/best_model.pkl", path_in_repo="best_model.pkl", repo_id=MODEL_REPO, repo_type="model")
api.upload_file(path_or_fileobj="models/scaler.pkl", path_in_repo="scaler.pkl", repo_id=MODEL_REPO, repo_type="model")
print("Model registered to Hugging Face Hub")
