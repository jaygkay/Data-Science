# Starter file for implementing significance weighting in Surprise
import numpy as np
from surprise import KNNWithMeans
from surprise import Dataset, Reader
from surprise.model_selection import PredefinedKFold
import unittest

class KNNSigWeighting(KNNWithMeans):
    """A KNN algorithm augmented with significance weighting

    The similarity between two users is adjusted by multiplying by the quantity:

    .. math::
    \\frac{
      min(|I_u \\cap I_v|, \\beta)}
      {\\beta}

    See Page 38 of the textbook
    
    """
    def __init__(self, k=40, min_k=1, sim_options=None, **kwargs):
       
        if sim_options is None:
            sim_options = {}
        KNNWithMeans.__init__(self, sim_options=sim_options, **kwargs)
        self.k = k
        self.min_k = min_k
        self.overlap = None
    
    def fit(self, trainset):
        """Model fitting for KNN with significance weighting

        Calls the parent class fit method and then generates the overlap matrix
        needed by the significance weighting.

        :param trainset:
        :return: self
        """
        # Call parent class function
        KNNWithMeans.fit(self, trainset)
        
        # Create an "overlap" matrix counting the number of items that
        # pairs of users have in common.
        # See the creation of the "freq" matrix in the "similarities.pyx" file.

        # Use overlap matrix to update the sim matrix, discounting by the significance weight factor.
        from six import iteritems
        
        n_x = trainset.n_users
        self.freq = np.zeros((n_x, n_x), np.int)
        
        for ui,item_rate1 in iteritems(trainset.ur):
            for uj,item_rate2 in iteritems(trainset.ur):
                if ui == uj:
                    self.freq[ui,uj]=0
                else:
                    for item1, rate1 in item_rate1:
                        for item2, rate2 in item_rate2:
                            if item1 == item2:
                                self.freq[ui,uj] += 1
        
        self.overlap = self.freq

        for i in range(n_x):
        	for j in range(n_x):
        		if (i>j):
        			self.sim[i,j] = self.sim[i,j] * self.sig_weight(i,j)
        			self.sim[j,i] = self.sim[i,j]

        
        return self

    def sig_weight(self, x1, x2):
        """Computes significance weight based on overlap and threshold.
        Threshold is provided in sim_options with key 'corate_threshold' with a default of 50.
        .. math::
           \\frac{
           min(|I_u \\cap I_v|, \\beta)}
           {\\beta}
        
        :param x1: user u
        :param x2: user v
        :return: the weight associated with the users x1 and x2
        """
        
        beta = self.sim_options['corate_threshold']
        weight = min(self.overlap[x1][x2], beta)/beta
        

        return weight

class TestSigWeight(unittest.TestCase):
    """Test case for significance weighting

    Don't run the experiments until these tests succeed.

    """

    def setUp(self):
        self.alg = KNNSigWeighting(sim_options={'name': 'pearson', 'user_based': True, 'corate_threshold': 25})
        self.trainset, self.testset = self.load_test_files()
        # self.show_trainset()

    def show_trainset(self):
        for uid, ratings in self.trainset.ur.items():
            user = self.trainset.to_raw_uid(uid)
            print('User {} ({})'.format(user, uid))
            for iid, rating in ratings:
                item = self.trainset.to_raw_iid(iid)
                print("{} ({}) = {}".format(item, iid, rating))

    def load_test_files(self):
        reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
        train_file = 'test-data-train.csv'
        test_file = 'test-data-test.csv'
        folds_files = [(train_file, test_file)]

        data = Dataset.load_from_folds(folds_files, reader=reader)
        pkf = PredefinedKFold()

        trainset, testset = next(pkf.split(data))

        return trainset, testset

    def test_data_load(self):
        self.assertEqual(len(self.testset), 5, "Test set not loaded correctly")

    def test_overlap_matrix(self):
        self.alg.fit(self.trainset)
        # These values should be 3
        threes = [(0, 1), (1, 2), (2, 3), (3, 4)]
        for u1, u2 in threes:
            self.assertEqual(self.alg.overlap[self.trainset.to_inner_uid(str(u1)),
                                              self.trainset.to_inner_uid(str(u2))],
                             3, "Overlap matrix incorrect at {} {}".format(u1, u2))

    def test_corate_fn(self):
        self.alg.fit(self.trainset)
        frac = self.alg.sig_weight(self.trainset.to_inner_uid('0'),
                                    self.trainset.to_inner_uid('1'))
        self.assertEqual(frac, 3.0/25, "Wrong co-rate fraction.")


if __name__ == '__main__':
    # Leave this for debugging
    # suite = unittest.TestSuite()
    # suite.addTest(TestCorate("test_overlap_matrix"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
    unittest.main()
