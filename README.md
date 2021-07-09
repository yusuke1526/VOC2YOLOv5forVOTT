# voc2yolov5

voc2yolov5 converts Pascal VOC data format into YOLOv5 data format. \
This program processes labels made using VOTT, otherwise it may not convert correctly.

Source directory have to be in below style.
```bash
.
├── Annotations
│   ├── 0000.xml
│   ├── 0001.xml
├── ImageSets
│   └── Main
│       ├── label0_train.txt
│       ├── label0_val.txt
│       ├── label1_train.txt
│       └── label1_val.txt
├── JPEGImages
│   ├── 0000.jpg
│   ├── 0001.jpg
└── pascal_label_map.pbtxt
```
VOTT would make this structure by default when you select Poscal VOC format.

# Usage
1. Install requirements.
```bash
pip install -r requirements.txt
```
2. Create target directory
```bash
mkdir yolo
```
3. Run voc2yolov5
```bash
python voc2yolov5.py --source_dir path/to/source/directory --target_dir path/to/target/directory
```
Then, YOLOv5 format dataset is in target directory, and data.yaml is in the current directory.

4. Edit data.yaml
add path, train, val, test (optional) information to data.yaml like below.
```yaml
path: data/mask  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images  # val images (relative to 'path')
# test: test/images

# following lines written by voc2yolov5.py
nc: 2
names:
- label0
- label1
```
