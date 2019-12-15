import pandas as pd
import matplotlib.pyplot as plt
from surprise import KNNBasic
from surprise import Dataset
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate
from surprise.model_selection import PredefinedKFold
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse
from surprise import Reader
from surprise import KNNWithMeans

import warnings
warnings.filterwarnings('ignore')

# 1. User_based with Pearson
data = Dataset.load_builtin('ml-100k')

cv = KFold(n_splits = 5, random_state = 30)
user = {'name':'pearson_baseline', 'user_based':True}
algo_user = KNNWithMeans(sim_options = user)
rmse_user = []

for trainset, testset in cv.split(data):
#for trainset, testset in pkf.split(data):
    algo_user.fit(trainset)
    pred_user = algo_user.test(testset)
    rmse_user.append(rmse(pred_user))
    rmse(pred_user)
rmse_user


# 2. ITEM_BASED with COSINE¶
item = {'name':'cosine', 'user_based':False}
algo_item = KNNWithMeans(sim_options = item)
rmse_item = []
for trainset, testset in cv.split(data):
    algo_item.fit(trainset)
    pred_item = algo_item.test(testset)
    rmse(pred_item)
    rmse_item.append(rmse(pred_item))

rmse_item


# 3. Summary 
# paired t-test
from scipy import stats
ttest = stats.ttest_rel(rmse_user, rmse_item)
res = []
for i in ttest:
    res.append(i)
# rmse for user and item
name = ['User_based','Item_based']
df = pd.DataFrame([rmse_user, rmse_item], index = name)


# 4. Results
from termcolor import colored
title1 = colored("<RMSE for User_based and Item_based>", "blue", attrs = ["bold"])
title2 = colored("<Significance of the deffience between User_based and Item_based>", 
                 "blue", attrs = ["bold"])
title3 = colored("<Paired T-statistics Results>", "blue", attrs = ["bold"])
print(title1)
#print colored("RMSE for User_based and Item_based", "blue")
print(df)
print()
print(title2)
results = [rmse_item, rmse_user]
box = pd.DataFrame(results, index = ['Item_based','User_based'])
box.T.boxplot(vert = False)
plt.show()
print()
print(title3)
print("Paired T-stat", res[0])
print("Paired P-value", res[1])
