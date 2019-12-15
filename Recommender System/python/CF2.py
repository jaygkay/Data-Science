import pandas as pd
import scipy as sp
from scipy.stats import ttest_rel
import numpy as np
from matplotlib import pyplot as plt
import heapq
import warnings
warnings.filterwarnings('ignore')

from surprise import KNNWithMeans
from surprise import Reader, Dataset
from surprise.model_selection import cross_validate, KFold, ShuffleSplit
from surprise.prediction_algorithms import PredictionImpossible

%load_ext autoreload
%autoreload
from sigweight1 import KNNSigWeighting
from prec_recall import pr_eval, precision_recall_at_k


# 1. Load the data
import os
file_path = os.path.expanduser('~/Desktop/filmtrust.txt')
reader = Reader(rating_scale = (0.5, 4))
#reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)
data.raw_ratings


#2. Convert the data to a data frame¶
col = ['User','Item','Rating','Unused']
df = pd.DataFrame(data.raw_ratings, columns = col)
print(df.shape)
df.head()

#3. Group by Item
group_item = df.groupby('Item')['Rating'].count()
group_item.head(10)

#4. group by user
group_user = df.groupby('User')['Rating'].count()
group_user.head(10)

#5. Knn with Means
from surprise.accuracy import rmse 
cv = KFold(n_splits = 5, random_state = 1,shuffle=False)
sim_opt = {'name':'pearson', 'user_based':True}
algo1 = KNNWithMeans(sim_options = sim_opt)
output1 = cross_validate(algo1, data, measures=['RMSE'], cv=cv, verbose=True)

# Significance weighting with corate_threshold = 100
sim_opt2 = {'name':'pearson', 'user_based':True,'corate_threshold':100}
algo2 = KNNSigWeighting(sim_options = sim_opt2)
output2 = cross_validate(algo2, data, measures=['RMSE'], cv=cv, verbose=True)

# Significance weighting with corate_threshold = 50
sim_opt3 = {'name':'pearson', 'user_based':True,'corate_threshold':50}
algo3 = KNNSigWeighting(sim_options = sim_opt3)
output3 = cross_validate(algo3, data, measures=['RMSE'], cv=cv, verbose=True)

rmse = [output1['test_rmse'], output2['test_rmse'], output3['test_rmse']]
plt.boxplot(rmse, labels=["KNNWithMeans", "KNNSigWeighting(100)", "KNNSigWeighting(50)"], vert=False, )
plt.show

# 6. Evaluation
output4 = pr_eval(algo1, data, cv, n= 10, threshold = 3.3)
pre_rec = [output4.T['Precision'], output4.T['Recall']]
plt.boxplot(pre_rec, labels=["Precision","Recall"], vert=False, )
plt.xticks([0.0000, 0.0010, 0.0020, 0.0030, 0.0040, 0.0050])

output5 = pr_eval(algo2, data, cv, n= 10, threshold = 3.3)
pre_rec = [output5.T['Precision'], output5.T['Recall']]
plt.boxplot(pre_rec, labels=["Precision","Recall"], vert=False, )
plt.xticks([0.0000, 0.0010, 0.0020, 0.0030, 0.0040, 0.0050])

output6 = pr_eval(algo3, data, cv, n= 10, threshold = 3.3)
pre_rec = [output6.T['Precision'], output6.T['Recall']]
plt.boxplot(pre_rec, labels=["Precision","Recall"], vert=False, )
plt.xticks([0.0000, 0.0010, 0.0020, 0.0030, 0.0040, 0.0050])

df1 = pd.DataFrame(output4.T, columns = ['Precision', 'Recall'])
df2 = pd.DataFrame(output5.T, columns = ['Precision', 'Recall'])
df3 = pd.DataFrame(output6.T, columns = ['Precision', 'Recall'])

ax1 = df1.plot(kind = 'scatter', x = 'Recall', y = 'Precision', color = 'b', label = 'KNNWithMeans') 
ax2 = df2.plot(kind = 'scatter', x = 'Recall', y = 'Precision', color = 'g', label = 'Sig100',ax = ax1) 
ax3 = df3.plot(kind = 'scatter', x = 'Recall', y = 'Precision', color = 'r', label = 'Sig50',ax = ax1) 

ax1.set_xlim(0,0.005)
ax1.set_ylim(0,0.005)


ax1.legend()
ax2.legend()
ax3.legend()
