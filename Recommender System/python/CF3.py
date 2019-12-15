import pandas as pd
import matplotlib.pyplot as plt
from surprise import KNNBasic
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate
from surprise.model_selection import PredefinedKFold
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse
from surprise import Reader
from surprise import KNNWithMeans
from surprise.model_selection import GridSearchCV
import timeit
import os
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv("review.csv")
reader = Reader(rating_scale = (1,5))
data = Dataset.load_from_df(df, reader = reader)


# 1. One neighborhood algorithm:
from surprise import KNNBaseline

# Setting the cross validation value
cv = KFold(n_splits = 5, random_state = 1,shuffle=False)

# Setting the grid search range for the parameters
param_grid = {'k': list(range(10,100,10)),
              'sim_options': {'name': ['pearson'],
                              'user_based': [True]}}
# Modeling with grid search
gs_knn = GridSearchCV(KNNBaseline, param_grid, measures = ['RMSE'], cv = cv)

# Fitting the data
gs_knn.fit(data)

# Assigning the all the values from the model into a dataframe
df_knn = pd.DataFrame.from_dict(gs_knn.cv_results)

# Selecting only 'parameter' and '5-cv-folds' columns from the result-dataframe
cols = ['params','split0_test_rmse','split1_test_rmse','split2_test_rmse','split3_test_rmse','split4_test_rmse']

# The data frame with the columns
df_knn[cols]

# The lowest rmse from the result
print("The best RMSE", df_knn[cols].min().min())

# The best parameters
print("The best parameters",gs_knn.best_params)



# "count" is used to find the n_th of row where the best parameter is stored in "df_knn[cols]"
count = 0
print("====================================================================================================")

# This For Loop is designed to match between 
# (the best parameter) and (the iteration for the parameter columns in df_knn[cols]).
# The "count" indicates number of iterations
for i in list(df_knn[cols].params):
    if [i] == list(gs_knn.best_params.values()):
        count += 1
        print("1. The best parameter is at", count ,"th row in the dataframe")


# The codes below are to extract only "best rmse" values
only_rmse = ['split0_test_rmse','split1_test_rmse','split2_test_rmse','split3_test_rmse','split4_test_rmse']
best_rmse_knn = pd.DataFrame(df_knn[only_rmse])
best_rmse_knn = best_rmse_knn.iloc[count]
print("====================================================================================================")
print("2. 5-fold RMSEs for the best parameter are as follow")
best_rmse_knn


# 2. Two model-based algorithms:

from surprise import SVD

param_grid = {'n_epochs': list(range(10,50,10)),
              'lr_all': [0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009],
              'reg_all':[0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]}
gs_svd = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=cv)
gs_svd.fit(data)
df_svd = pd.DataFrame.from_dict(gs_svd.cv_results)
cols = ['params','split0_test_rmse','split1_test_rmse','split2_test_rmse','split3_test_rmse','split4_test_rmse']
df_svd[cols]

print("The best RMSE", df_svd[cols].min().min())
print("The best parameters",gs_svd.best_params)

count = 0
print("====================================================================================================")
for i in list(df_svd[cols].params):
    if [i] == list(gs_svd.best_params.values()):
        print("1. The best parameter is :", i)
        count += 1
only_rmse = ['split0_test_rmse','split1_test_rmse','split2_test_rmse','split3_test_rmse','split4_test_rmse']
best_rmse_svd = pd.DataFrame(df_svd[only_rmse])
best_rmse_svd = best_rmse_svd.iloc[count]
print("====================================================================================================")
print("2. 5-fold RMSEs for the best parameter are as follow")
best_rmse_svd

# 3. SVD ++
from surprise import SVDpp

param_grid = {'n_epochs': list(range(10,50,10)),
              'lr_all': [0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009],
              'reg_all':[0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]}

gs_svdpp = GridSearchCV(SVDpp, param_grid, measures=['rmse'], cv=cv)
gs_svdpp.fit(data)
df_svdpp = pd.DataFrame.from_dict(gs_svdpp.cv_results)
cols = ['params','split0_test_rmse','split1_test_rmse','split2_test_rmse','split3_test_rmse','split4_test_rmse']
df_svdpp[cols]

print("The best RMSE", df_svdpp[cols].min().min())
print("The best parameters",gs_svdpp.best_params)

count = 0
print("====================================================================================================")
for i in list(df_svdpp[cols].params):
    if [i] == list(gs_svdpp.best_params.values()):
        print("1. The best parameter is :", i)
        count += 1
only_rmse = ['split0_test_rmse','split1_test_rmse','split2_test_rmse','split3_test_rmse','split4_test_rmse']
best_rmse_svdpp = pd.DataFrame(df_svdpp[only_rmse])
best_rmse_svdpp = best_rmse_svdpp.iloc[count]
print("====================================================================================================")
print("2. 5-fold RMSEs for the best parameter are as follow")
best_rmse_svdpp

# 4. Plot 5-folds-CV for each model
rmse_res = [best_rmse_knn, best_rmse_svd, best_rmse_svdpp]
plt.boxplot(rmse_res, labels = ['KNNBaseline','SVD','SVD++'],vert = False)
plt.show()
