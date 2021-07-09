import os
import yaml
import shutil
import argparse
import xml.etree.ElementTree as ET

def xml_reader(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    size = tree.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        objects.append(obj_struct)
    return width, height, objects


def voc2yolo(img_name, place):
    img_path = f"{args.source_dir}/JPEGImages/{img_name}"
    xml_path = img_path.replace("JPEGImages", "Annotations").replace(".jpg", ".xml")
    width, height, objects = xml_reader(xml_path)

    lines = []
    for obj in objects:
        x, y, x2, y2 = obj['bbox']
        class_name = obj['name']
        label = classes_dict[class_name]
        cx = (x2+x)*0.5 / width
        cy = (y2+y)*0.5 / height
        w = (x2-x)*1. / width
        h = (y2-y)*1. / height
        line = "%d %.6f %.6f %.6f %.6f\n" % (int(label)-1, cx, cy, w, h)
        lines.append(line)

    img_path_copy = img_path.replace(args.source_dir, args.target_dir).replace('JPEGImages', 'images').split('/')
    img_path_copy.insert(2, place)
    img_path_copy = '/'.join(img_path_copy)
    print(img_path)
    print(img_path_copy)
    txt_path = xml_path.replace(args.source_dir, args.target_dir).replace('Annotations', 'labels').split('/')
    txt_path.insert(2, place)
    txt_path = '/'.join(txt_path).replace(".xml", ".txt")
    print(xml_path)
    print(txt_path)
    shutil.copy(img_path, img_path_copy)
    with open(txt_path, "w") as f:
        f.writelines(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--target_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(f'{args.target_dir}/train/images', exist_ok=True)
    os.makedirs(f'{args.target_dir}/train/labels', exist_ok=True)
    os.makedirs(f'{args.target_dir}/val/images', exist_ok=True)
    os.makedirs(f'{args.target_dir}/val/labels', exist_ok=True)

    class_names = []
    classes_dict = {}
    with open(f"{args.source_dir}/pascal_label_map.pbtxt") as f:
        for line in f.readlines():
            l = line.strip()
            if ('{' in l) or ('}' in l):
                continue
            if 'id' in l:
                id = int(l.split(':')[-1])
            elif 'name' in l:
                class_name = l.split("'")[-2]
                class_names.append(class_name)
                classes_dict[class_name] = id

    # data.yaml
    with open("data.yaml", "w") as f:
        yaml.safe_dump({
            'nc': len(class_names),
            'names': class_names,
        }, f, sort_keys=False)

    # train
    train_img_name_list = []
    with open(f"{args.source_dir}/ImageSets/Main/{class_names[0]}_train.txt") as f:
        for line in f.readlines():
            file_name = line.strip().split(" ")[0]
            train_img_name_list.append(file_name)
    for img_name in train_img_name_list:
        voc2yolo(img_name, "train")

    # val
    val_img_name_list = []
    with open(f"{args.source_dir}/ImageSets/Main/{class_names[0]}_val.txt") as f:
        for line in f.readlines():
            file_name = line.strip().split(" ")[0]
            val_img_name_list.append(file_name)
    for img_name in val_img_name_list:
        voc2yolo(img_name, "val")