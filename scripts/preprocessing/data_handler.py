import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit

class DataHandler:
    """
    A class to handle data preprocessing tasks such as loading data, chunking posts, 
    removing short posts, downsampling authors, and splitting data into train, validation, 
    and test sets.

    Attributes:
        file_path (str): The path to the CSV file containing the data.
        data (pd.DataFrame): The DataFrame containing the loaded data.

    Methods:
        get_data(): Returns the DataFrame containing the data.
        add_word_count(column='post'): Adds a word count column to the DataFrame.
        chunk_posts(column='post', word_limit=300): Chunks posts into pieces with at most `word_limit` words.
        remove_short_posts(min_word_count=300): Removes posts with a word count less than `min_word_count`.
        downsample_authors(n=500): Downsamples the data to include only `n` posts per author.
        split_data(train_size=0.7, test_size=0.15, group_col='author_ID', random_state=42): Splits the data into train, validation, and test sets.
    """
    def __init__(self, file_path):
        """
        Initialize the DataHandler with the path to the CSV file containing the data.
        """
        self.file_path = file_path
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"Loading data from {file_path}...")
        self.data = pd.read_csv(file_path)

        self.data.rename(columns={self.data.columns[0]: 'author_ID'}, inplace=True)
        self.add_word_count()
        
        print("Data loaded successfully.")


    def get_data(self):
        """
        Returns the DataFrame containing the data.
        """
        return self.data
    
    
    def add_word_count(self, column='post'):
        """
        Add a column to the DataFrame containing the word count of the specified column.
        """
        self.data['word_count'] = self.data[column].apply(lambda x: len(x.split()))


    def chunk_posts(self, column='post', word_limit=300):
        """
        Chunk the instances of the DataFrame into pieces with at most `word_limit` words in the specified column.

        Parameters:
            column (str): The column to chunk based on word count. Default is 'post'.
            word_limit (int): The maximum number of words per chunk. Default is 300.

        Returns:
            None: The method modifies the DataFrame in place.
        """
        chunked_rows = []

        for _, row in self.data.iterrows():
            text = row[column]
            words = text.split()

            # Split the text into chunks of at most `word_limit` words
            for i in range(0, len(words), word_limit):
                chunk = ' '.join(words[i:i + word_limit])
                new_row = row.copy()
                new_row[column] = chunk
                chunked_rows.append(new_row)

        # Modify the DataFrame in place
        self.data = pd.DataFrame(chunked_rows)
        self.add_word_count()


    def remove_short_posts(self, min_word_count=300):
        """
        Remove posts with a word count less than `min_word_count`.

        Parameters:
            min_word_count (int): The minimum number of words required for a post to be kept. Default is 300.

        Returns:
            None: The method modifies the DataFrame in place.
        """
        self.data = self.data[self.data['word_count'] >= min_word_count]
        self.data.drop(columns=['word_count'], inplace=True)


    def downsample_authors(self, n=500):
        """
        Downsample the data to include only `n` posts per author.

        Parameters:
            n (int): The number of posts to keep per author. Default is 100.

        Returns:
            None: The method modifies the DataFrame in place.
        """
        self.data = self.data.groupby('author_ID').head(n)


    def get_split_data(self, train_size=0.7, test_size=0.15, group_col='author_ID', random_state=42):
        """
        Splits the dataset into train, validation, and test sets based on group labels such that each group is only present in one set.
        
        Parameters:
            df (pd.DataFrame): The input dataset with samples and group labels.
            train_size (float): Proportion of data to include in the training set.
            test_size (float): Proportion of data to include in the test set.
            group_col (str): The column containing group labels (e.g., 'author_id').
            random_state (int): Random state for reproducibility.
        
        Returns:
            train_df (pd.DataFrame): Training subset.
            val_df (pd.DataFrame): Validation subset.
            test_df (pd.DataFrame): Testing subset.
        """
        assert train_size + test_size <= 1.0, "Train and test sizes must sum to less than or equal to 1.0"
        
        # Extract groups
        groups = self.data[group_col].values

        # First split: train-test and validation
        val_size = 1.0 - (train_size + test_size)
        gss1 = GroupShuffleSplit(n_splits=1, train_size=(train_size + test_size), random_state=random_state)
        train_test_idx, val_idx = next(gss1.split(self.data, groups=groups))
        train_test_df, val_df = self.data.iloc[train_test_idx], self.data.iloc[val_idx]

        # Second split: train and test
        gss2 = GroupShuffleSplit(n_splits=1, train_size=(train_size / (train_size + test_size)), random_state=random_state)
        train_idx, test_idx = next(gss2.split(train_test_df, groups=train_test_df[group_col].values))
        train_df, test_df = train_test_df.iloc[train_idx], train_test_df.iloc[test_idx]

        return train_df, val_df, test_df