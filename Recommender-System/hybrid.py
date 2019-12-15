""" Starting file for implementing weighted hybrid in Surprise"""

import numpy as np
from surprise import KNNWithMeans, AlgoBase
from surprise import Dataset, Reader, Prediction
from surprise.model_selection import PredefinedKFold
from sklearn.linear_model import LinearRegression
from fixed import FixedPredictor
import unittest

class WeightedHybrid(AlgoBase):

    _components = None
    _weights = None
    trainset = None

    def __init__(self, components):
        """ Constructor for WeightedHybrid

        :param components: The list of components to include in the hybrid
        :param normalize: If True, then weights are normalized to sum to 1 when set.
        """
        AlgoBase.__init__(self)

        # Set instance variables
        self._components = components



    def set_weights(self, weights):
        """ Set the hybrid weights and normalize.

        :param weights: New weights
        :return: No value
        """
        self._weights = weights
        self.normalize_weights()   ########################added
        #self.get_weights()
        

    def normalize_weights(self):
        """ Normalize weight vector.
        Negative weights set to zero, and whole vector sums to 1.0.
        :return: No value
        """
        
        # Set negative weights to zero
        # Normalize to sum to one.
        


        self.new_weight=[]
        for i in self._weights:
            if any(i < 0 for i in self._weights):
                self.new_weight = [0,1]

            elif all(i == 0 for i in self._weights):
                i = 1/len(self._weights)
                self.new_weight.append(i)
            else:
                i = i/sum(self._weights)
                self.new_weight.append(i)

        # If the weights are all zeros, set weights equal to 1/k, where k is the number
        # of components.
        self._weights = self.new_weight
        self._weights = np.round(self._weights,3)
       

    def get_weights(self):
        return self._weights

    def fit(self, trainset):
        """ Fitting procedure for weighted hybrid.
        :param trainset:
        :return: No value
        """
        # Set the trainset instance variable
        self.trainset = trainset

        # Fit all of the components using the trainset.
        for comp in self._components:
            comp.fit(self.trainset)

        # Create arrays for call to LinearRegression function        
        
        ######### creat x array and y array for Linear Regression
        x_arr = []
        y_arr = []
        #u_id = []
        #i_id = []
        #rat_id = []
        for uid, iid, rating in self.trainset.all_ratings():
            #u_id.append(uid)
            #i_id.append(iid)
            #rat_id.append(rating)
            pred_arr = []
            for comp in self._components:
                pred = comp.estimate(uid, iid)
                if type(pred) != tuple:
                	pred_arr.append(pred)
                else:
                	pred_arr.append(pred[0])
                #pred_arr.append(pred[0])
            x_arr.append(pred_arr)
            y_arr.append(rating)
        
        #print("uid", u_id)
        #print("iid", i_id)
        #print("rating", rat_id)
        #print("The shape of x array", np.array(x_arr).shape)
        #print("The shape of y array", np.array(y_arr).shape)
          # One array has dimensions [r, k] where r is the number of ratings and k is the number of components.
        # The array has the predictions for all the u,i pairs for all the components.
        # The other array has dimensions [r, 1] has all the ground truth rating values.
        # Do not clip the predicted rating values.
        
        
        # Compute the LinearRegression.
        LR = LinearRegression(fit_intercept=True, copy_X = False)
        # fit_intercept=True, because the data is not zero-centered
        # copy_X = False for efficiency, we don't need the arrays for anything else.
       
        LR.fit(x_arr, y_arr)

        # Set the weights.
        print("The weight before normalizing",LR.coef_)
        self.set_weights(LR.coef_)
        #print(self.set_weights(LR.coef_))
        # For debugging purposes, show the learned weights.
        print("Learned weights {}".format(self.get_weights()))
        


    def predict(self, uid, iid, r_ui=None, clip=True, verbose=False):
        """ Predict a rating using the hybrid.

        :param uid: User id
        :param iid: Item id
        :param r_ui: Observed rating
        :param clip: If True, clip to rating range
        :param verbose:
        :return: A Prediction object
        """

        # Iterate over the components and build up an aggregate score using the weights.
        # Clip the estimate into [lower_bound, higher_bound] if clip parameter is True. See algo_base.py.
        # Create a new Prediction object. Retain any Prediction "details" values
        # in a new dictionary.
        try:
            iuid = self.trainset.to_inner_uid(uid)
        except ValueError:
            iuid = 'UKN__' + str(uid)
        try:
            iiid = self.trainset.to_inner_iid(iid)
        except ValueError:
            iiid = 'UKN__' + str(iid)     
        
        details = {}
        try:
            cnt = 0
            est = 0.0
            for comp in self._components:
                pred = comp.estimate(iuid, iiid)
                if type(pred) == tuple:
                	pred = pred[0]
                	
                est += pred[0] * self._weights[cnt]
                cnt += 1        
    
            # If the details dict was also returned
            if isinstance(est, tuple):
                est, details = est

            details['was_impossible'] = False

        except:
            est = self.trainset.global_mean
            details['was_impossible'] = True
            details['reason'] = str('PredictionImpossible')
        

# clip estimate into [lower_bound, higher_bound]
        if clip:
            lower_bound, higher_bound = self.trainset.rating_scale
            est = min(higher_bound, est)
            est = max(lower_bound, est)

        pred = Prediction(uid, iid, r_ui, est, details)

        if verbose:
            print(pred)

        return pred
class TestHybrid(unittest.TestCase):
    """Test case for significance weighting

    Don't run the experiments until these tests succeed.

    """

    def setUp(self):
        np.random.seed(2018)
        comp1 = FixedPredictor(rating=3.0)
        comp2 = KNNWithMeans(k=3, min_k=1, sim_options={'name': 'cosine', 'user_based': True})
        self.alg1 = WeightedHybrid([comp1, comp2])
        self.alg2 = WeightedHybrid([comp1, comp2])
        self.alg3 = WeightedHybrid([comp1, comp2])
        self.alg4 = WeightedHybrid([comp1, comp1, comp1, comp1])
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
        # This time, we'll use the built-in reader.
        reader = Reader(line_format='user item rating', sep=',', skip_lines=1)

        # folds_files is a list of tuples containing file paths:
        # [(u1.base, u1.test), (u2.base, u2.test), ... (u5.base, u5.test)]
        train_file = 'test-data-train.csv'
        test_file = 'test-data-test.csv'
        folds_files = [(train_file, test_file)]

        data = Dataset.load_from_folds(folds_files, reader=reader)
        pkf = PredefinedKFold()

        trainset, testset = next(pkf.split(data))

        # print("Users")
        # for i in range(0, 5):
        #     print ('raw: {} inner: {}'.format(i, trainset.to_inner_uid(str(i))))
        #
        # print("Items")
        # for i in range(0, 5):
        #     print ('raw: {} inner: {}'.format(i, trainset.to_inner_iid(str(i))))

        return trainset, testset

    def test_data_load(self):
        self.assertEqual(len(self.testset), 5, "Test set not loaded correctly")

    def test_combine(self):
        trainset = self.trainset

        self.alg1.fit(trainset)
        self.alg2.fit(trainset)
        self.alg3.fit(trainset)

        # ignore learned weights
        self.alg1.set_weights([0,1])
        self.alg2.set_weights([1,0])
        self.alg3.set_weights([0.5, 0.5])



        uid = '1'
        iid = '1'

        p1 = self.alg1.predict(uid, iid, 3.0)
        p2 = self.alg2.predict(uid, iid, 3.0)
        p3 = self.alg3.predict(uid, iid, 3.0)

        self.assertAlmostEqual(p3.est, p1.est*0.5 + p2.est*0.5,
                         msg="Combination incorrect: {} != 0.5*{}+0.5*{}".format(p3.est, p1.est, p2.est))

    def test_regress(self):
        self.alg1.fit(self.trainset)

        weights = self.alg1.get_weights()
        self.assertAlmostEqual(weights[0], 0, msg="Fitted weight for constant predictor not zero.")

        self.alg4.fit(self.trainset)

        weights = self.alg4.get_weights()

        self.assertAlmostEqual(weights[0], 0.25, msg="Fitted weights for matching predictor should be 1/n.")

        self.assertAlmostEqual(weights[0], weights[1], msg="Fitted weights for matching predictors not equal.")

    def test_details(self):
        self.alg1.fit(self.trainset)
        bad_pred = self.alg1.predict('100', '100', 3.0)

        self.assertTrue(isinstance(bad_pred.details, dict), "No details dictionary returned")

        self.assertTrue('Comp1' in bad_pred.details, "Missing details entry for component")

        self.assertTrue(bad_pred.details['Comp1']['was_impossible'],
                        "Details entry incorrect. User and Item should be unknown.")

    def test_normalize(self):
        self.alg1.fit(self.trainset)

        self.assertAlmostEqual(np.sum(self.alg1.get_weights()), 1.0, msg="Weights not normalized")


if __name__ == '__main__':
    # Leave this for debugging
    # suite = unittest.TestSuite()
    # suite.addTest(TestHybrid("test_combine"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
    unittest.main()
