

# CV Project: Real-time Human Detection

## Installation

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Datasets

### Dataset Structure

Ensure your datasets are structured as follows:

```
data
├── images
│   ├── train
│   │   ├── train0.jpg
│   │   └── train1.jpg
│   └── val
│       ├── val0.jpg
│       └── val1.jpg
├── labels
│   ├── train
│   │   ├── train0.txt
│   │   └── train1.txt
│   └── val
│       ├── val0.txt
│       └── val1.txt
└── GT
    ├── train
    │   ├── train0.txt
    │   └── train1.txt
    └── val
        ├── val0.txt
        └── val1.txt
```

The dataset combines ECP, CH, and Indoor datasets, balanced and sampled appropriately. The `images` folder contains images, `labels` contains soft labels, and `GT` contains annotations.

### Dataset Preparation

#### Indoor Dataset

- Download the dataset from [our Google Drive link](https://drive.google.com/drive/folders/1hc1ufnAEfkDUpORWIqVkmAkW2YeBu5rN). It's already preprocessed.
  - Download `cv_indoor_data.zip` from the `indoor_data_and_softlabel` folder.


#### CrowdHuman Dataset

- Download train images and annotations from [CrowdHuman official link](https://www.crowdhuman.org/download.html).
- Download YOLO formatted soft labels from [our Google Drive link](https://drive.google.com/drive/folders/1hc1ufnAEfkDUpORWIqVkmAkW2YeBu5rN). Download `crowdhuman_yolo_labels.zip`.
- Convert annotations to YOLO format using `src/convert_CH_GT.py`.
- Balance the dataset using `src/sampling_code/sample_data_CH.py`.
- Convert soft labels for YOLO compatibility using `src/preprocessing_label.py`.

#### Eurocity Persons Dataset

- Download day images and annotations from [ECP official link](https://eurocity-dataset.tudelft.nl/eval/downloads/detection#).
- Download YOLO formatted soft labels from [our Google Drive link](https://drive.google.com/drive/folders/1hc1ufnAEfkDUpORWIqVkmAkW2YeBu5rN). Download all files from the 'ECP' folder.
- Combine the first 59 data from the Amsterdam, Barcelona, Basel, Berlin, and Bologna folders in the original train set of the ECP dataset with the entire validation set of the ECP dataset to form a new training set.
- Construct the new validation set using the first 16 files from each folder in the original train set, starting from Bratislava to Wuerzburg.
- Convert new train and validation set annotations from JSON to YOLO format using `src/convert_ECP_GT.py`.
- Convert soft labels for YOLO compatibility using `src/preprocessing_label.py`.

### Dataset Location

Place the `data` folder alongside the `YOLO_v6` directory:

```
Parent dir
├── data
│
└── YOLO_v6
```

### Inference Images

Place the inference images as follows:

```
Parent dir
├── data
│
└── YOLO_v6
   │
   └── infer_images
```

### YAML File

Place your YAML file in `YOLO_v6/data/`. For example, `my_dataset.yaml` should be located at `YOLO_v6/data/my_dataset.yaml`.

```yaml
# Example YAML content

# COCO 2017 dataset <http://cocodataset.org>
train: ../data/images/train 
val: ../data/images/val  
# test: ./data/coco/images/test2017 (option)

# number of classes
nc: 1
# whether it is coco dataset, only coco dataset should be set to True.
is_coco: False

# class names
names: [ 'person']
```

## Running the Code
First, you'll need to enter the YOLOv6 folder.
```bash
cd YOLOv6
```


### Running the Baseline

```bash
python tools/train.py --batch 64 --conf configs/yolov6s_finetune.py --data data/my_dataset.yaml --fuse_ab --device 0 --specific-shape --width 448 --height 448 --epochs 10
```

### Running MTKD Method

```bash
python tools/train.py --batch 64 --conf configs/yolov6s_finetune.py --data data/my_dataset.yaml --MTKD --device 0 --specific-shape --width 448 --height 448 --epochs 10
```

### Running Inference

```bash
python tools/infer.py --weights runs/train/exp/weights/best_ckpt.pt --source images --device 0 --save-txt
```

### Converting TXT to COCO Format

```
python src/coco2json.py
```

## Utilities

### Resizing Input Images

```bash
python resize_input.py -d <dataset_type>
# dataset_type: CH, ECP, IN
```

### Visualizing Bounding Boxes

```bash
python src/visualize_bbox.py -d <dataset_type> -s <score>
# dataset_type: CH, ECP, IN
# score: threshold (default: 0.5)
```

