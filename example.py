from google.cloud import storage
client = storage.Client(project='mod-gcp-white-soi-dev-1')
bucket = client.bucket('civilian-benchmark-datasets')
print([blob for blob in bucket.list_blobs(prefix='hit-uav')])