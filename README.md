# parquet-dataloder
PyTorch DataLoader to read larger than memory files.

`parquet_dataset.py` contains the PyTorch Dataset implementation of ParquetDataset.
`test_dataset.py` contains test to validate ParquetDataset.
Run `pip install -r requirements.txt` to install requirements.
Run `pytest test_dataset.py -v` to test the dataset functionality.

Limitations of `ParquetDataset`:
- Batch Size shouldn't exceed the available memory
- Parallel data loading (e.g., with num_workers) is not included here and would require additional logic.
- This implementation assumes the entire batch to be converted into a PyTorch tensor. Custom preprocessing through `collate_fn` might be necessary for more complex datasets.
