import unittest

class TestAnalysisImports(unittest.TestCase):

    def test_imports(self):
        # Confirm that sklearn, umap, and shap are imported without errors
        import sklearn
        import umap
        import shap
        import pandas as pd
        import numpy as np
        
        self.assertIsNotNone(sklearn)
        self.assertIsNotNone(umap)
        self.assertIsNotNone(shap)

if __name__ == "__main__":
    unittest.main()
