from google.cloud import storage
from pathlib import Path
from ThermalClassifier.datasets import datasets_data
import os

def download_dataset(root_dir, dataset_name):
    if os.path.exists(f"{root_dir}/{dataset_name}"):
        return
    
    bucket_name = datasets_data[dataset_name]['BUCKET_NAME']
    dataset_dir = datasets_data[dataset_name]['DATASET_DIR']


    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=dataset_dir)  # Get list of files
    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        file_split = blob.name.split("/")
        file_name = file_split[-1]
        dataset_name_index = file_split.index(dataset_name)
        relative_dir = Path("/".join(file_split[dataset_name_index:-1]))
        final_file_local_path = root_dir/relative_dir/file_name
        if final_file_local_path.exists():
            continue
        (root_dir/relative_dir).mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(final_file_local_path)