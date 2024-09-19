import unittest
from database import DataProcessor

class TestUtils(unittest.TestCase):
    """
    A class to perform unit tests on the DataProcessor class.

    Methods:
    -------
    setUp():
        Sets up the DataProcessor instance and loads the data before each test.
    test_load_csv_to_df():
        Tests if the CSV file is loaded into a DataFrame and is not empty.
    test_process_test_data():
        Tests if the test data is processed and results are not empty.
    """
    def setUp(self):
        """
        Sets up the DataProcessor instance and loads the data before each test.

        This method is called before each test case to initialize the DataProcessor
        with the training, ideal functions, and test data files. It also creates the
        database and loads the data into it.
        """
        self.processor = DataProcessor(
            training_file='train.csv',
            ideal_functions_file='ideal.csv',
            test_file='test.csv'
        )
        self.processor.create_database()
        self.processor.load_data()

    def test_load_csv_to_df(self):
        """
        Tests if the CSV file is loaded into a DataFrame and is not empty.

        This test case checks whether the 'load_csv_to_df' method successfully loads
        the training data CSV file into a pandas DataFrame and verifies that the
        DataFrame is not empty.
        """
        df = self.processor.load_csv_to_df('train.csv')
        self.assertFalse(df.empty)

    def test_process_test_data(self):
        """
        Tests if the test data is processed and results are not empty.

        This test case checks whether the 'process_test_data' method successfully
        processes the test data and verifies that the resulting DataFrame is not empty.
        """
        results = self.processor.process_test_data()
        self.assertFalse(results.empty)

if __name__ == "__main__":
    unittest.main()
