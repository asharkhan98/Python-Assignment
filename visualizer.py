import pandas as pd
from bokeh.plotting import figure, show
from bokeh.io import output_file
from bokeh.layouts import gridplot
import sqlalchemy as db

class DataVisualizer:
    """
    A class to visualize training data, ideal functions, and test data using Bokeh.

    Attributes:
    ----------
    db_file : str
        Path to the SQLite database file (default is 'data.db').

    Methods:
    -------
    visualize_data():
        Visualizes the training data, ideal functions, and test data.
    """
    def __init__(self, db_file='data.db'):
        """
        Constructs all the necessary attributes for the DataVisualizer object.

        Parameters:
        ----------
        db_file : str
            Path to the SQLite database file (default is 'data.db').
        """
        self.db_file = db_file

    def visualize_data(self):
        """
        Visualizes the training data, ideal functions, and test data using Bokeh.

        This method reads the data from the SQLite database, creates scatter plots
        for the training data and corresponding ideal functions, and visualizes the
        test data along with the assigned ideal functions. The resulting plots are
        displayed in a grid layout in an HTML file.
        """
        engine = db.create_engine(f'sqlite:///{self.db_file}')
        output_file("visualization.html")

        # Load data from the database
        training_data = pd.read_sql('SELECT * FROM training_data', engine)
        ideal_functions = pd.read_sql('SELECT * FROM ideal_functions', engine)
        test_data = pd.read_sql('SELECT * FROM test_data', engine)

        plots = []

        # Create scatter plots for training data and ideal functions
        for i in range(1, 5):
            p = figure(title=f'Training Data Y{i} and Ideal Function', x_axis_label='x', y_axis_label='y')
            p.scatter(training_data['x'], training_data[f'y{i}'], legend_label=f'Training y{i}', color='blue')
            p.line(ideal_functions['x'], ideal_functions[f'y{i}'], legend_label=f'Ideal y{i}', color='red')
            plots.append(p)

        # Create scatter plot for test data with assigned ideal functions
        p = figure(title='Test Data with Assigned Ideal Functions', x_axis_label='x', y_axis_label='Y')
        p.scatter(test_data['x'], test_data['y'], legend_label='Test Data', color='green')

        for i in range(1, 5):
            subset = test_data[test_data['ideal_func_no'] == i]
            p.scatter(subset['x'], subset['y'], legend_label=f'Ideal Func {i}', color='blue')

        # Arrange the plots in a grid layout and display them
        grid = gridplot([[plots[0], plots[1]], [plots[2], plots[3]], [p]])
        show(grid)
