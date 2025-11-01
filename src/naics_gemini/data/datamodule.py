import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from .dataset import NaicsDataset
import pandas as pd

class NaicsDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for the NAICS dataset.
    It handles loading data, creating datasets, and setting up dataloaders.
    """
    def __init__(self, data_path: str, tokenizer, 
                 batch_size: int = 32, max_seq_length: int = 512,
                 val_split_size: float = 0.1, test_split_size: float = 0.1,
                 num_workers: int = 4, seed: int = 42):
        """
        Args:
            data_path (str): Path to the CSV file containing the data.
                             Expected columns: 'text_column_name' and 'label_column_name'.
            tokenizer: A Hugging Face style tokenizer.
            batch_size (int): The batch size for the dataloaders.
            max_seq_length (int): The maximum sequence length for tokenization.
            val_split_size (float): The proportion of data to use for validation.
            test_split_size (float): The proportion of data to use for testing.
            num_workers (int): Number of workers for the dataloader.
            seed (int): Random seed for reproducible splits.
        """
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.val_split_size = val_split_size
        self.test_split_size = test_split_size
        self.num_workers = num_workers
        self.seed = seed
        
        # Placeholders for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['tokenizer'])

    def prepare_data(self):
        """
        Called once on one GPU. 
        Use this to download or pre-process data.
        """
        # In this case, we just assume the data_path is a local CSV.
        # If it were a download, it would happen here.
        pass

    def setup(self, stage: str = None):
        """
        Called on every GPU.
        This is where we load data, split it, and create PyTorch Datasets.
        
        Args:
            stage (str, optional): 'fit', 'validate', 'test', or 'predict'.
                                   Used to control which datasets are set up.
        """
        
        # Load the full dataset
        # TODO: Update column names to match your CSV
        full_data = pd.read_csv(self.data_path)
        texts = full_data['text_column_name'].tolist()
        
        # TODO: Ensure your labels are integers (e.g., by using LabelEncoder)
        labels = full_data['label_column_name'].astype(int).tolist()

        full_dataset = NaicsDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length
        )

        # Split into train, val, test
        total_size = len(full_dataset)
        test_size = int(total_size * self.test_split_size)
        val_size = int(total_size * self.val_split_size)
        train_size = total_size - val_size - test_size
        
        # Check for rounding errors
        if train_size + val_size + test_size != total_size:
            train_size += (total_size - (train_size + val_size + test_size))

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self):
        """Returns the DataLoader for the training set."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Returns the DataLoader for the validation set."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Returns the DataLoader for the test set."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
