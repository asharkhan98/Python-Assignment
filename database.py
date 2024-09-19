import pandas as pd
import sqlalchemy as db
from sqlalchemy.orm import sessionmaker
import numpy as np
from sklearn.metrics import mean_squared_error

class DataLoadingException(Exception):
    """
    Custom exception for errors encountered during data loading.
    """
    pass

class DataProcessor:
    """
    A class to process data for training, ideal functions, and test datasets.

    Attributes:
    ----------
    training_file : str
        Path to the training data CSV file.
    ideal_functions_file : str
        Path to the ideal functions CSV file.
    test_file : str
        Path to the test data CSV file.
    db_file : str
        Path to the SQLite database file (default is 'data.db').

    Methods:
    -------
    load_csv_to_df(file):
        Loads a CSV file into a pandas DataFrame.
    create_database():
        Creates SQLite database tables for training data, ideal functions, and test data.
    load_data():
        Loads training and ideal functions data into the database.
    process_test_data():
        Processes the test data and maps it to the best fitting ideal functions.
    """
    def __init__(self, training_file, ideal_functions_file, test_file, db_file='data.db'):
        """
        Constructs all the necessary attributes for the DataProcessor object.

        Parameters:
        ----------
        training_file : str
            Path to the training data CSV file.
        ideal_functions_file : str
            Path to the ideal functions CSV file.
        test_file : str
            Path to the test data CSV file.
        db_file : str
            Path to the SQLite database file (default is 'data.db').
        """
        self.training_file = training_file
        self.ideal_functions_file = ideal_functions_file
        self.test_file = test_file
        self.db_file = db_file

    def load_csv_to_df(self, file):
        """
        Loads a CSV file into a pandas DataFrame and converts column names to lowercase.

        Parameters:
        ----------
        file : str
            Path to the CSV file.

        Returns:
        -------
        DataFrame
            A pandas DataFrame containing the loaded data.
        
        Raises:
        ------
        DataLoadingException
            If there is an error loading the CSV file.
        """
        try:
            df = pd.read_csv(file)
            df.columns = [col.lower() for col in df.columns]  # Convert all column names to lowercase
            print(f"Loaded data from {file}:")
            print(df.head())  # Print the first few rows of the DataFrame for debugging
            print(f"Columns: {df.columns.tolist()}")  # Print column names for debugging
            return df
        except Exception as e:
            raise DataLoadingException(f"Error loading {file}: {str(e)}")

    def create_database(self):
        """
        Creates SQLite database tables for training data, ideal functions, and test data.
        """
        engine = db.create_engine(f'sqlite:///{self.db_file}')
        Session = sessionmaker(bind=engine)
        session = Session()
        metadata = db.MetaData()

        # Create training data table
        training_data = db.Table('training_data', metadata,
                                 db.Column('x', db.Float),
                                 db.Column('y1', db.Float),
                                 db.Column('y2', db.Float),
                                 db.Column('y3', db.Float),
                                 db.Column('y4', db.Float))

        # Create ideal functions table
        ideal_functions = db.Table('ideal_functions', metadata,
                                   db.Column('x', db.Float),
                                   *(db.Column(f'y{i+1}', db.Float) for i in range(50)))

        # Create test data table
        test_data = db.Table('test_data', metadata,
                             db.Column('x', db.Float),
                             db.Column('y', db.Float),
                             db.Column('delta_y', db.Float),
                             db.Column('ideal_func_no', db.Integer))

        metadata.create_all(engine)
        self.engine = engine
        self.training_data_table = training_data
        self.ideal_functions_table = ideal_functions
        self.test_data_table = test_data
        self.session = session

    def load_data(self):
        """
        Loads training and ideal functions data into the database.
        """
        # Load training data
        training_data = self.load_csv_to_df(self.training_file)
        training_data.columns = ['x', 'y1', 'y2', 'y3', 'y4']

        # Load ideal functions data
        ideal_functions_data = self.load_csv_to_df(self.ideal_functions_file)

        # Insert into database
        training_data.to_sql('training_data', self.engine, if_exists='replace', index=False)
        ideal_functions_data.to_sql('ideal_functions', self.engine, if_exists='replace', index=False)

    def process_test_data(self):
        """
        Processes the test data and maps it to the best fitting ideal functions.

        Returns:
        -------
        DataFrame
            A pandas DataFrame containing the test data with the corresponding ideal function and deviation.
        """
        test_data = self.load_csv_to_df(self.test_file)

        # Retrieve the ideal functions
        ideal_functions_df = pd.read_sql('SELECT * FROM ideal_functions', self.engine)
        print("Ideal functions data loaded from the database:")
        print(ideal_functions_df.head())  # Print the first few rows for debugging
        print(f"Columns: {ideal_functions_df.columns.tolist()}")  # Print column names for debugging
        ideal_functions_df.set_index('x', inplace=True)

        training_data = pd.read_sql('SELECT * FROM training_data', self.engine)

        # Determine the best fit for each training function
        best_ideal_funcs = []
        for i in range(1, 5):
            training_col = f'y{i}'
            min_mse = float('inf')
            best_func = None
            for col in ideal_functions_df.columns:
                mse = mean_squared_error(training_data[training_col], ideal_functions_df[col])
                if mse < min_mse:
                    min_mse = mse
                    best_func = col
            best_ideal_funcs.append(best_func)

        # Process each test data point
        results = []
        for _, row in test_data.iterrows():
            x_val = row['x']
            y_val = row['y']
            best_fit = None
            min_deviation = float('inf')
            for idx, ideal_func in enumerate(best_ideal_funcs):
                deviation = abs(y_val - ideal_functions_df.loc[x_val, ideal_func])
                max_allowed_deviation = np.sqrt(2) * abs(training_data[f'y{idx+1}'] - ideal_functions_df[ideal_func]).max()
                if deviation <= max_allowed_deviation and deviation < min_deviation:
                    min_deviation = deviation
                    best_fit = idx + 1
            results.append({'x': x_val, 'y': y_val, 'delta_y': min_deviation, 'ideal_func_no': best_fit})

        results_df = pd.DataFrame(results)
        results_df.to_sql('test_data', self.engine, if_exists='replace', index=False)
        return results_df
