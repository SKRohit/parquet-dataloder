import torch
from torch.utils.data import IterableDataset
import pyarrow.parquet as pq


class ParquetDataset(IterableDataset):
    """
    A PyTorch IterableDataset for streaming data from larger-than-memory Parquet files.
    """
    def __init__(self, parquet_file, batch_size=1000):
        """
        Initialize the dataset.
        
        Args:
            parquet_file (str): Path to the Parquet file.
            batch_size (int): Number of rows to read at a time.
        """
        self.parquet_file = parquet_file
        self.batch_size = batch_size

    def __iter__(self):
        """
        Iterate over the Parquet file in batches.
        """
        reader = pq.ParquetFile(self.parquet_file)
        # Read a batch of rows
        for table in reader.iter_batches(batch_size=self.batch_size):
            batch_data = table.to_pandas().to_numpy()
            yield torch.tensor(batch_data)