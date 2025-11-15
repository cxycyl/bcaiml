# Quick Start

The following is a detailed description of this project.

## 1.Environment Configuration

#### 1.1. First, create and activate a conda virtual environment

```bash
conda create -n bcaiml python==3.10
conda activate bcaiml
```

#### 1.2. Install required packages

```bash
pip install -r requirements.txt
```

## 2.Test

#### 2.1. Prepare Datasets

The test set is specified in the `test_datasets.json` file. If the data set is in `json` format, then the dataset is placed in the path specified by the contents of the corresponding json file. If it is a folder path, create two subfolders, `Tp` and `Gt`, within that folder, where `Tp` will hold the image and `Gt` will hold the corresponding mask.The "Negative" is the all-black mask, which is the fully real image, so there is no need to enter path.

Below is the `test_datasets.json` file.

```
{
    "Columbia": "./data/datasets/Columbia.json",
    "coverage": "./data/datasets/coverage.json",
    "coverage": "/data/datasets/coverage.json",
    "CASIAv1": "/data/datasets/CASIA1.0"
}
```

#### 2.2. Standard F1

```bash
bash test_f1.sh
```

#### 2.3. Permute F1

```bash
bash test_permute_f1.sh
```

#### 2.4. AUC and IoU

```bash
bash test_auc_iou.sh
```
#### 2.5. Robust test

```
bash test_robust.sh
```

## 3.Training
#### 3.1. Prepare Datasets

The training set is specified in the `balanced_datasets.json` file, where for each dataset in the `JsonDataset` format, the dataset is placed in the path specified by the contents of the corresponding json file; If it is in `ManiDataset` format, then this corresponds to a folder path with two subfolders, `Tp` and `Gt`, where `Tp` holds the image and `Gt` holds the mask.

Below is the `balanced_datasets.json` file.

```
[
    [
        "JsonDataset",
        "./data/datasets/FantasticReality_v1/FantasticReality.json"
    ],
    [
        "ManiDataset",
        "./data/datasets/CASIA2.0"
    ],
    [
        "ManiDataset",
        "./data/datasets/IMD_20_1024"
    ],
    [
        "JsonDataset",
        "./data/datasets/compRAISE/compRAISE_1024_list.json"
    ],
    [
        "JsonDataset",
        "./data/datasets/tampCOCO/sp_COCO_list.json"
    ],
    [
        "JsonDataset",
        "./data/datasets/tampCOCO/cm_COCO_list.json"
    ],
    [
        "JsonDataset",
        "./data/datasets/tampCOCO/bcm_COCO_list.json"
    ],
    [
        "JsonDataset",
        "./data/datasets/tampCOCO/bcmc_COCO_list.json"
    ]
]
```

#### 3.2. Segformer Pretrained File Download

To begin the training process, you need to download the pretrained weights for Segformer. Specifically, this project uses the **mit-b3** model pretrained on ImageNet. Follow the instructions below to download it from the official Segformer GitHub repository:

Visit the official Segformer GitHub repository: [Segformer GitHub](https://github.com/NVlabs/SegFormer). Navigate to the **"Training"** section in the repository's README or directly access the download link provided for the **mit-b3** model. Download the pretrained weights for **mit-b3**.

#### 3.3. Configure Parameters

Before running the training shell script, edit and configure the following parameters in the `.sh` file as needed:

- **`seg_pretrain_path`**: This should point to the pretrained segmentation model file. Ensure the file exists at the specified location.

  ```bash
  seg_pretrain_path="/bca_iml/data/segformer/mit_b3.pth"
  ```
  
- **`data_path`**: This is the directory containing the training data. Update this path to the location of your training dataset

  ```bash
  data_path="/bca_iml/balanced_dataset.json"
  ```
  
- **`test_data_path`**: This is the directory containing the testing data. Update this path to the location of your test dataset.

  ```bash
  test_data_path="/bca_iml/data/datasets/CASIA1.0"
  ```

#### 3.4. **Run the Training Script**

Once the parameters are correctly configured, execute the shell script to start the training process. Use the following command:

```bash
bash train.sh
```

**Note:** If you experience connectivity issues with Huggingface, you can set the following environment variable to use an alternative endpoint:

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```
