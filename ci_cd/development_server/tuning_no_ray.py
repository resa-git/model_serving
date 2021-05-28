
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPRegressor
#from sklearn. import SVC

from sklearn.model_selection import cross_validate, train_test_split
from sklearn.datasets import fetch_covtype
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
import numpy as np

#Dataset here
data_file = 'model_data2'
df = pd.read_pickle(data_file)
df = df.drop("fork", axis=1)
df = df.sample(frac=1)
scaler = StandardScaler()
scale_columns = ["size", "forks", "open_issues_count"]
df[scale_columns] = scaler.fit_transform(df[scale_columns])
df = pd.DataFrame(df, columns=df.columns)
test_df = df[-10:]

df = df.iloc[:-10]
#test.to_csv("test.csv")
df.to_csv("train.csv")
test_df.to_csv("test.csv")
y = df["stargazers_count"]
X = df.drop("stargazers_count", axis=1)

print(X)
print(y)

rfc = RandomForestRegressor(n_estimators = 150, max_depth = 50, random_state=0)
max_rfc = cross_validate(rfc, X,y,scoring="r2")["test_score"].mean()

mlp_model = MLPRegressor(activation="relu", solver="adam", learning_rate="constant", max_iter=200)
max_ada = cross_validate(mlp_model, X,y,scoring="r2")["test_score"].mean()

ada_model = AdaBoostRegressor(n_estimators=50, loss="linear")
max_mlp = cross_validate(ada_model, X,y,scoring="r2")["test_score"].mean()

linear_model = LinearRegression(normalize=True, fit_intercept=True)
max_linear = cross_validate(linear_model, X,y,scoring="r2")["test_score"].mean()

#df_rfc = analysis_rfc.results_df
#df_ada = analysis_ada.results_df
#df_mlp = analysis_mlp.results_df
#df_linear = analysis_linear.results_df

#max_rfc = df_rfc["score"].max()
#print("Max for rfc: "+ str(max_rfc))
#max_ada = df_ada["score"].max()
#print("Max for ada: "+ str(max_ada))
#max_mlp = df_mlp["score"].max()
#print("Max for mlp: "+ str(max_mlp))
#max_linear = df_linear["score"].max()
#print("Max for linear: "+ str(max_linear))
max_list = [max_rfc,max_mlp, max_ada, max_linear]

model = None

if max_rfc == max(max_list):
    model = rfc
elif max_ada == max(max_list):
    model = ada_model
elif max_mlp == max(max_list):
    model = mlp_model
elif max_linear == max(max_list):
    model = linear_model

print("The best model had params")
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=42)
model.fit(X_train,y_train)
pred = model.predict(X_test)
print(y_test)
print(pred)
score = r2_score(y_test, pred)
print(score)
#create this folder if it does not exist
filename = "models/final.sav"
pickle.dump(model, open(filename, 'wb'))

