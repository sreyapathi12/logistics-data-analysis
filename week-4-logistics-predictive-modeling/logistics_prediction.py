from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT=Path(__file__).resolve().parent
df=pd.read_csv(ROOT/"data"/"logistics_hypothetical_dataset.csv")
out=ROOT/"outputs"; out.mkdir(exist_ok=True)
features=["Distance_km","Shipment_Weight_kg","Transport_Mode","Region"]; target="Delivery_Time_days"
df=df.dropna(subset=features+[target]); X=df[features]; y=df[target]
prep=ColumnTransformer([
 ("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),["Distance_km","Shipment_Weight_kg"]),
 ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),["Transport_Mode","Region"])
])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
cv=KFold(n_splits=5,shuffle=True,random_state=42)
results=[]
for name,est in [("Linear Regression",LinearRegression()),("Random Forest",RandomForestRegressor(n_estimators=200,min_samples_leaf=2,random_state=42))]:
 pipe=Pipeline([("prep",prep),("model",est)]); pipe.fit(Xtr,ytr); p=pipe.predict(Xte)
 results.append({"Model":name,"MAE":mean_absolute_error(yte,p),"RMSE":np.sqrt(mean_squared_error(yte,p)),"R2":r2_score(yte,p),"CV_RMSE":-cross_val_score(pipe,Xtr,ytr,cv=cv,scoring="neg_root_mean_squared_error").mean()})
rf=Pipeline([("prep",prep),("model",RandomForestRegressor(random_state=42))])
gs=GridSearchCV(rf,{"model__n_estimators":[100,200],"model__max_depth":[None,8],"model__min_samples_leaf":[1,2,4]},cv=cv,scoring="neg_root_mean_squared_error",n_jobs=-1)
gs.fit(Xtr,ytr); p=gs.predict(Xte)
results.append({"Model":"Tuned Random Forest","MAE":mean_absolute_error(yte,p),"RMSE":np.sqrt(mean_squared_error(yte,p)),"R2":r2_score(yte,p),"CV_RMSE":-gs.best_score_})
pd.DataFrame(results).to_csv(out/"model_evaluation_metrics.csv",index=False)
pd.DataFrame({"Actual":yte.to_numpy(),"Predicted":p,"Absolute_Error":abs(yte.to_numpy()-p)}).to_csv(out/"test_set_predictions.csv",index=False)
print(pd.DataFrame(results).round(3).to_string(index=False))
plt.figure(figsize=(6,4)); plt.scatter(yte,p,alpha=.7); lo=min(yte.min(),p.min()); hi=max(yte.max(),p.max()); plt.plot([lo,hi],[lo,hi],"--"); plt.xlabel("Actual days"); plt.ylabel("Predicted days"); plt.title("Actual vs Predicted"); plt.tight_layout(); plt.savefig(out/"actual_vs_predicted.png",dpi=150); plt.show(); plt.close()
plt.figure(figsize=(7,4)); plt.bar([r["Model"] for r in results],[r["RMSE"] for r in results]); plt.ylabel("RMSE (days)"); plt.title("Model Comparison"); plt.xticks(rotation=18,ha="right"); plt.tight_layout(); plt.savefig(out/"model_comparison_rmse.png",dpi=150); plt.show(); plt.close()
