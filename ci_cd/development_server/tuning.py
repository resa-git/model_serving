
import ray
from ray import tune
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
#X,y = make_regression(n_samples=1000, n_features=10, shuffle=True, random_state=1)

def random_forest_training(config):
    n_estimators, max_depth, ccp_alpha = config["n_estimators"], config["max_depth"], config["ccp_alpha"]
    rfc = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, ccp_alpha = ccp_alpha, random_state=0)
    score = cross_validate(rfc, X,y,scoring="r2")["test_score"].mean()
    tune.report(score=score)


def ada_training(config):
    n_estimators , loss = config["n_estimators"], config["loss"]
    ada_model = AdaBoostRegressor(n_estimators=n_estimators, loss=loss)
    score = cross_validate(ada_model, X,y,scoring="r2")["test_score"].mean()
    tune.report(score=score)

def mlp_training(config):
    activation,solver,learning_rate,max_iter = config["activation"], config["solver"], config["learning_rate"],config["max_iter"]
    mlp_model = MLPRegressor(activation=activation, solver=solver, learning_rate=learning_rate, max_iter=max_iter)
    score = cross_validate(mlp_model, X,y,scoring="r2")["test_score"].mean()
    tune.report(score=score)

def linear_training(config):
    normalize, fit_intercept= config["normalize"], config["fit_intercept"]
    linear_model = LinearRegression(normalize=normalize, fit_intercept=fit_intercept)
    score = cross_validate(linear_model, X,y,scoring="r2")["test_score"].mean()
    tune.report(score=score)


ray.init(address='172.17.0.2:6379', _redis_password='5241590000000000')


analysis_rfc = tune.run(
    random_forest_training,
    config={
        "n_estimators": tune.grid_search([50,75,100,150]),
        "max_depth": tune.grid_search([5,10,50,100]),
        "ccp_alpha": tune.grid_search([0, 0.005, 0.015, 0.03])
    }, metric="score", mode="max")


analysis_ada = tune.run(
    ada_training,
    config={
        "n_estimators": tune.grid_search([10, 50, 100]),
        "loss": tune.grid_search(["linear", "square", "exponential"])
    }, metric="score", mode="max")

analysis_mlp = tune.run(
    mlp_training,
    config={
        "activation": tune.grid_search(["identity", "logistic", "relu"]),
        "solver": tune.grid_search(["lbfgs", "adam"]),
        "learning_rate": tune.grid_search(["constant", "adaptive"]),
        "max_iter": tune.grid_search([200,500])
    }, metric="score", mode="max")

analysis_linear = tune.run(
    linear_training,
    config={
        "fit_intercept": tune.grid_search([True,False]),
        "normalize": tune.grid_search([True, False])
    }, metric="score", mode="max")

df_rfc = analysis_rfc.results_df
df_ada = analysis_ada.results_df
df_mlp = analysis_mlp.results_df
df_linear = analysis_linear.results_df

max_rfc = df_rfc["score"].max()
print("Max for rfc: "+ str(max_rfc))
max_ada = df_ada["score"].max()
print("Max for ada: "+ str(max_ada))
max_mlp = df_mlp["score"].max()
print("Max for mlp: "+ str(max_mlp))
max_linear = df_linear["score"].max()
print("Max for linear: "+ str(max_linear))
max_list = [max_rfc,max_mlp, max_ada, max_linear]

model = None
config = None

if max_rfc == max(max_list):
    config = analysis_rfc.get_best_config(metric="score", mode="max")
    model = RandomForestRegressor(**config)
elif max_ada == max(max_list):
    config = analysis_ada.get_best_config(metric="score", mode="max")
    model = AdaBoostRegressor(**config)    
elif max_mlp == max(max_list):
    config = analysis_mlp.get_best_config(metric="score", mode="max")
    model = MLPRegressor(**config)
elif max_linear == max(max_list):
    config = analysis_linear.get_best_config(metric="score", mode="max")
    model = LinearRegression(**config)

print("The best model had params" + str(config))
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=42)
model.fit(X_train,y_train)
pred = model.predict(X_test)
print(y_test)
print(pred)
score = r2_score(y_test, pred)
print(score)
#create this folder if it does not exist
filename = "/home/appuser/jump/final.sav"
pickle.dump(model, open(filename, 'wb'))

