
import os, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from huggingface_hub import HfApi, login
import joblib

HF_TOKEN = os.environ["HF_TOKEN"]
HF_USERNAME = "harshverma27"
DATASET_REPO = f"{HF_USERNAME}/engine-sensor-data"

login(token=HF_TOKEN, add_to_git_credential=False)
api = HfApi()

FEATURE_COLS = ["Engine rpm","Lub oil pressure","Fuel pressure",
                "Coolant pressure","lub oil temp","Coolant temp"]
TARGET_COL = "Engine Condition"

HF_RAW_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/data/engine_data.csv"
df = pd.read_csv(HF_RAW_URL)
df.columns = [c.strip() for c in df.columns]
df.drop_duplicates(inplace=True)
df[TARGET_COL] = df[TARGET_COL].astype(int)

X, y = df[FEATURE_COLS], df[TARGET_COL]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)
pd.DataFrame(X_train_sc, columns=FEATURE_COLS).assign(**{TARGET_COL: y_train.values}).to_csv("data/processed/train.csv", index=False)
pd.DataFrame(X_test_sc, columns=FEATURE_COLS).assign(**{TARGET_COL: y_test.values}).to_csv("data/processed/test.csv", index=False)
joblib.dump(scaler, "models/scaler.pkl")

api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", exist_ok=True)
for local, remote in [("data/processed/train.csv","data/train.csv"),
                       ("data/processed/test.csv","data/test.csv")]:
    api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=DATASET_REPO, repo_type="dataset")
print("Data prep complete, uploaded to Hub")
