## ThermalClassifier
Creating a Bbox classifier 

## Datasets
This repo has some assumptions for the various datasets:

1. Each dataset need to be in a seperate folder
2. In each dataset folder there need to be .json file for the annotations.
3. The annotations need to be in COCO format !!!
4. If you want to donwload dataset from the gcp. Check if the dataset is in datasets_data dict in the datasets/__init__.py.

## Usage
This code is managed by .yaml files in the configs folder.
If you have any request to upgrade this repo please add an issuse :)

* NOTE: In the .yaml file the dataset name needs to be "dataset_folder/.json"
```
test_datasets: ['hit-uav/test.json']
```

## How to debug with vscode IDE
To use --updated with vscode launch.json you need to send the string like this:
```
"args": [
        "--updates", "{\"epochs\": 10}"
]
```

