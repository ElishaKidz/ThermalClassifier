from google.cloud import storage
from pathlib import Path
from ThermalClassifier.datasets import datasets_data


def download_dataset(root_dir, dataset_name):
    bucket_name = datasets_data[dataset_name]['BUCKET_NAME']
    dataset_name = datasets_data[dataset_name]['DATASET_NAME']

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=dataset_name)  # Get list of files
    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        file_split = blob.name.split("/")
        file_name = file_split[-1]
        relative_dir = Path("/".join(file_split[0:-1]))
        final_file_local_path = root_dir/relative_dir/file_name
        if final_file_local_path.exists():
            continue
        (root_dir/relative_dir).mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(final_file_local_path)