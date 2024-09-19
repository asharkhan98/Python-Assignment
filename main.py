from database import DataProcessor
from visualizer import DataVisualizer

def main():
    """
    Main function to run the data processing and visualization.

    This function initializes the DataProcessor with paths to the training data,
    ideal functions, and test data CSV files. It then creates the database, loads
    the data, and processes the test data. After processing, it initializes the
    DataVisualizer to visualize the data and displays the results.
    """
    # Initialize DataProcessor with file paths
    processor = DataProcessor(
        training_file='train.csv',
        ideal_functions_file='ideal.csv',
        test_file='test.csv'
    )

    # Create the database and load data
    processor.create_database()
    processor.load_data()

    # Process the test data
    processor.process_test_data()

    # Initialize DataVisualizer and visualize data
    visualizer = DataVisualizer()
    visualizer.visualize_data()

if __name__ == "__main__":
    main()
