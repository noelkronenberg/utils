import unittest
import pandas as pd
import numpy as np
from utils.analysis import logit, lca

class TestAnalysisFunctions(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'outcome': np.random.randint(0, 2, size=100),
            'confounder_1': np.random.randn(100),
            'confounder_2': np.random.randn(100),
            'confounder_categorical': np.random.choice(['A', 'B', 'C'], size=100)
        })

    def test_logit_basic(self):
        result = logit(self.data, 'outcome', ['confounder_1', 'confounder_2'])
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'params'))

    def test_logit_with_categorical(self):
        result = logit(self.data, 'outcome', ['confounder_1', 'confounder_2', 'confounder_categorical'], categorical_vars=['confounder_categorical'])
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'params'))

    def test_logit_with_dropna(self):
        data_with_na = self.data.copy()
        data_with_na.loc[0:10, 'confounder_1'] = np.nan
        result = logit(data_with_na, 'outcome', ['confounder_1', 'confounder_2'], dropna=True)
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'params'))

    def test_lca_unsupervised(self):
        result = lca(self.data.drop('confounder_categorical', axis=1))
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'n_components'))

    def test_lca_supervised(self):
        result = lca(self.data, outcome='outcome', confounders=['confounder_1', 'confounder_2'])
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'n_components'))

if __name__ == '__main__':
    unittest.main()