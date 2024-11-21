import pytest
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
from parquet_dataset import ParquetDataset 

@pytest.fixture
def generate_parquet_file():
    """
    Fixture to generate synthetic Parquet files for testing.
    """
    def _generate(filename, num_rows, num_columns):
        data = {f"col{i}": np.random.rand(num_rows).astype(np.float32) for i in range(num_columns)}
        df = pd.DataFrame(data)
        df.to_parquet(filename)
        return filename
    return _generate


@pytest.mark.parametrize("num_rows, num_columns, batch_size", [
    (10_005, 5, 1000),  # Dataset with non-multiple of batch_size rows
    (0, 5, 1000),       # Empty dataset
    (1_000, 5, 100),    # Small dataset
])
def test_parquet_dataset(generate_parquet_file, num_rows, num_columns, batch_size):
    """
    Test that the ParquetDataset processes rows correctly.
    """
    filename = generate_parquet_file("test_data.parquet", num_rows, num_columns)
    dataset = ParquetDataset(filename, batch_size=batch_size)
    dataloader = DataLoader(dataset, batch_size=None, pin_memory=True)

    total_rows = 0
    for batch in dataloader:
        assert batch.ndim == 2, "Batch should be a 2D tensor (rows, columns)."
        assert batch.shape[1] == num_columns, "Number of columns in batch is incorrect."
        total_rows += batch.shape[0]

    assert total_rows == num_rows, f"Expected {num_rows} rows, but got {total_rows}."



def test_last_batch_size(generate_parquet_file):
    """
    Test that the last batch is processed correctly when its size is smaller than batch_size.
    """
    num_rows = 10_005
    num_columns = 6
    batch_size = 1000
    filename = generate_parquet_file("last_batch_test.parquet", num_rows, num_columns)

    dataset = ParquetDataset(filename, batch_size=batch_size)
    dataloader = DataLoader(dataset, batch_size=None, pin_memory=True)

    batches = list(dataloader)
    print(len(batches), batches[-1].shape)
    assert batches[-1].shape[0] == num_rows%batch_size, "Last batch row does not match column count."



@pytest.fixture(scope="function", autouse=True)
def cleanup_generated_files():
    """
    Fixture to clean up Parquet files after tests.
    """
    yield
    for file in ["test_data.parquet", "last_batch_test.parquet"]:
        if os.path.exists(file):
            os.remove(file)