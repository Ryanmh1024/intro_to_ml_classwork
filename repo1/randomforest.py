# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve,
    classification_report, RocCurveDisplay,
    accuracy_score
)

sns.set_style('whitegrid')

df = pd.read_csv("CellphoneAddiction.csv")
X = df.drop(columns=["addicted_label","transaction_id","user_id","addiction_level"])
y = df["addicted_label"]
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
transformer=ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', MinMaxScaler(), num_cols)
    ]
)
pipeline=Pipeline(steps=[
    ('preprocessor', transformer),
    ('model', RandomForestClassifier(n_jobs=-1, random_state=42, max_depth=5))
])

y_pred = pipeline.fit(X_train, y_train).predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(test_accuracy)
cross_val=cross_val_score(pipeline, X, y, cv=5)
print(cross_val)
prob = pipeline.predict_proba(X_test)
pred = pipeline.predict(X_test)
m = pd.DataFrame({'neg_prob':prob[:, 0], 'pred':pred, 'target':y_test, 'pos_prob':prob[:, 1]})
# %%
ConfusionMatrixDisplay.from_predictions(m.target, m.pred, display_labels=[False, True], colorbar=False)
RocCurveDisplay.from_predictions(m.target, m.pos_prob)
print(classification_report(y_test,y_pred))