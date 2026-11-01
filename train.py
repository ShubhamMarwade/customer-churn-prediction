"""Customer churn prediction: cleaning, feature engineering, SMOTE, model comparison, SHAP."""
import json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, shap
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, RocCurveDisplay
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")
SEED = 42

df = pd.read_csv("data/Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df["Churn"] = (df["Churn"] == "Yes").astype(int)
df = df.drop(columns="customerID")

# feature engineering
svc = ["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
df["num_services"] = sum((df[c] == "Yes").astype(int) for c in svc)
df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
df["tenure_group"] = pd.cut(df["tenure"], [-1,12,24,48,72], labels=["0-12","13-24","25-48","49-72"]).astype(str)
df["autopay"] = df["PaymentMethod"].str.contains("automatic").astype(int)

y = df.pop("Churn"); X = df
cat = X.select_dtypes("object").columns.tolist(); num = [c for c in X.columns if c not in cat]
pre = ColumnTransformer([("num", StandardScaler(), num),
                         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

models = {
 "Logistic Regression": LogisticRegression(max_iter=2000),
 "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
 "XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, eval_metric="logloss", random_state=SEED, n_jobs=4),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows, fitted = [], {}
for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("smote", SMOTE(random_state=SEED)), ("clf", m)])
    cv_auc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc").mean()
    pipe.fit(Xtr, ytr); p = pipe.predict_proba(Xte)[:,1]; pred = (p >= .5).astype(int)
    rows.append(dict(model=name, cv_roc_auc=round(cv_auc,4), test_roc_auc=round(roc_auc_score(yte,p),4),
                     precision=round(precision_score(yte,pred),4), recall=round(recall_score(yte,pred),4),
                     f1=round(f1_score(yte,pred),4)))
    fitted[name] = pipe; print(rows[-1])
res = pd.DataFrame(rows); res.to_csv("outputs/model_comparison.csv", index=False)

best = res.sort_values("test_roc_auc", ascending=False).iloc[0]["model"]
pipe = fitted[best]; p = pipe.predict_proba(Xte)[:,1]
# threshold tuning for F1
ths = np.arange(.2,.8,.02); f1s = [f1_score(yte,(p>=t).astype(int)) for t in ths]
t_best = float(ths[int(np.argmax(f1s))])
json.dump({"best_model": best, "threshold": t_best, "test_roc_auc": float(roc_auc_score(yte,p)),
           "f1_at_threshold": float(max(f1s)), "n_rows": int(len(df)), "churn_rate": float(y.mean())},
          open("outputs/metrics.json","w"), indent=2)
joblib.dump({"pipeline": pipe, "threshold": t_best, "columns": X.columns.tolist()}, "models/churn_model.joblib")

fig, ax = plt.subplots(figsize=(6,5))
for n, pp in fitted.items(): RocCurveDisplay.from_estimator(pp, Xte, yte, name=n, ax=ax)
ax.plot([0,1],[0,1],"k--",lw=.8); ax.set_title("ROC curves (hold-out set)"); plt.tight_layout()
plt.savefig("outputs/roc_curves.png", dpi=140); plt.close()

# SHAP on best (tree) model
if best != "Logistic Regression":
    Xt = pipe.named_steps["pre"].transform(Xte)
    names = pipe.named_steps["pre"].get_feature_names_out()
    names = [n.split("__",1)[1] for n in names]
    sv = shap.TreeExplainer(pipe.named_steps["clf"]).shap_values(Xt)
    if isinstance(sv, list): sv = sv[1]
    shap.summary_plot(sv, Xt, feature_names=names, show=False, max_display=12)
    plt.tight_layout(); plt.savefig("outputs/shap_summary.png", dpi=140, bbox_inches="tight"); plt.close()
    imp = pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False).head(10)
    imp.to_csv("outputs/top_churn_drivers.csv", header=["mean_abs_shap"]); print(imp)
print("best:", best, "threshold:", t_best, "F1:", max(f1s))
