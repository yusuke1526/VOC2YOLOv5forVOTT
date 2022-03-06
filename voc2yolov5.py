import os
import yaml
import shutil
import argparse
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
    img_path = os.path.join(args.source_dir, 'JPEGImages', img_name)
    xml_path = img_path.replace("JPEGImages", "Annotations").replace(".jpg", ".xml")
    width, height, objects = xml_reader(xml_path)

    lines = []
    
    for obj in objects:
        x, y, x2, y2 = obj['bbox']
        class_name = obj['name']
        if class_name in args.exclude_cls:
            continue
        label = classes_dict[class_name]
        cx = (x2+x)*0.5 / width
        cy = (y2+y)*0.5 / height
        w = (x2-x)*1. / width
        h = (y2-y)*1. / height
        line = "%d %.6f %.6f %.6f %.6f\n" % (int(label)-1, cx, cy, w, h)
        lines.append(line)
        class_counts[place][class_name] += 1

    img_path_copy = img_path.replace(args.source_dir, args.target_dir).replace('JPEGImages', 'images').split('/')
    img_path_copy.insert(2, place)
    img_path_copy = '/'.join(img_path_copy)
    # print(img_path)
    # print(img_path_copy)
    txt_path = xml_path.replace(args.source_dir, args.target_dir).replace('Annotations', 'labels').split('/')
    txt_path.insert(2, place)
    txt_path = '/'.join(txt_path).replace(".xml", ".txt")
    # print(xml_path)
    # print(txt_path)
    shutil.copy(img_path, img_path_copy)
    with open(txt_path, "w") as f:
        f.writelines(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', '-s', type=str, required=True)
    parser.add_argument('--target-dir', '-t', type=str, required=True)
    parser.add_argument('--train-size', type=int, default=7)
    parser.add_argument('--val-size', type=int, default=2)
    parser.add_argument('--test-size', type=int, default=1)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--exclude-cls', type=str, nargs='*', default = [])
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.target_dir, 'all/images'), exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, 'all/labels'), exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, 'train/images'), exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, 'train/labels'), exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, 'val/images'), exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, 'val/labels'), exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, 'test/images'), exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, 'test/labels'), exist_ok=True)

    class_names = []
    classes_dict = {}
    with open(os.path.join(args.source_dir, 'pascal_label_map.pbtxt')) as f:
        for line in f.readlines():
            l = line.strip()
            if ('{' in l) or ('}' in l):
                continue
            if 'id' in l:
                id = int(l.split(':')[-1])
            elif 'name' in l:
                class_name = l.split("'")[-2]
                if class_name in args.exclude_cls:
                    continue
                class_names.append(class_name)
                classes_dict[class_name] = id

    # data.yaml
    target_dir = os.path.basename(os.path.dirname(args.target_dir))
    print(args.target_dir)
    with open(os.path.join(args.target_dir, "data.yaml"), "w") as f:
        yaml.safe_dump({
            'path': os.path.join('./data', target_dir),
            'train': './train',
            'val': './val',
            'test': './test',
            'all': './all',
            'nc': len(class_names),
            'names': class_names,
        }, f, sort_keys=False)

    img_names_dict = {'all': [], 'train': [], 'val': [], 'test': []}
    with open(os.path.join(args.source_dir, 'ImageSets/Main/', f"{class_names[0]}.txt")) as f:
        for line in f.readlines():
            file_name = line.strip().split(" ")[0]
            img_names_dict['all'].append(file_name)

    img_names_dict['all'] = sorted(list(img_names_dict['all']))

    img_names_dict['train'], img_names_dict['val'] = train_test_split(img_names_dict['all'], shuffle=args.shuffle, test_size=(args.val_size + args.test_size)/(args.train_size + args.val_size + args.test_size), random_state=args.seed)
    if args.test_size > 0:
        img_names_dict['val'], img_names_dict['test'] = train_test_split(img_names_dict['val'], shuffle=args.shuffle, test_size=args.test_size/(args.val_size + args.test_size), random_state=args.seed)

    for key, item in img_names_dict.items():
        print(f"{key} data num: {len(img_names_dict[key])}")

    class_counts = {v: {name: 0 for name in class_names} for v in img_names_dict.keys()}

    for place, img_names in img_names_dict.items():
        for img_name in img_names:
            voc2yolo(img_name, place)

    for place in img_names_dict.keys():
        fig = plt.figure(figsize=(5, 2.5))
        plt.tight_layout
        plt.bar(class_counts[place].keys(), class_counts[place].values())
        plt.ylabel('instances')
        plt.savefig(os.path.join(args.target_dir, f'label_distribution_{place}.png'))
        print(place, class_counts[place])