from data_handler import DataHandler
import os

def main():
    # Load the raw data
    dir_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
    file_path = os.path.join(dir_path, 'political_leaning.csv')

    # Preprocess the data
    data_handler = DataHandler(file_path)
    data_handler.chunk_posts()
    data_handler.remove_short_posts()
    data_handler.downsample_authors()

    # Split the data into train, validation, and test sets
    train_df, val_df, test_df = data_handler.get_split_data()

    # Save the processed data
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', '500')
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print("Data preprocessing completed.")

if __name__ == "__main__":
    main()