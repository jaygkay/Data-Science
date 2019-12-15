
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from surprise import KNNWithMeans, SVD, SVDpp, CoClustering, NMF, SlopeOne, BaselineOnly
from surprise import Reader, Dataset
from surprise.model_selection import cross_validate, KFold
%load_ext autoreload
%autoreload
from hybrid import WeightedHybrid


data = Dataset.load_builtin('ml-100k')
fold5 = KFold(n_splits=5, random_state=1, shuffle=False)
algo1 = KNNWithMeans(k=2, sim_options={'name': 'cosine', 'user_based': True})
algo2 = KNNWithMeans(k=2, sim_options={'name': 'cosine', 'user_based': False})
hybrid = WeightedHybrid([algo1, algo2])

output1 = cross_validate(algo1, data, measures=['RMSE'], cv=fold5, verbose=True)
output2 = cross_validate(algo2, data, measures=['RMSE'], cv=fold5, verbose=True)
output3 = cross_validate(hybrid, data, measures=['RMSE'], cv=fold5, verbose=True)

rmse = [output1['test_rmse'], output2['test_rmse'],output3['test_rmse']]
plt.boxplot(rmse, labels = ['User_based','Item_baed','Hybrid'], vert = False)
plt.xticks([1.00, 1.05, 1.10, 1.15, 1.20])
plt.show()
