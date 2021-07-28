import csv
import json
from datetime import datetime
from os.path import join
from typing import Tuple

import cv2


def create_image_info(image_id: int, file_name: str, image_size: Tuple[int, int],
                      date_captured: datetime = datetime.utcnow().isoformat(' '),
                      license_id: int = 1, coco_url: str = "", flickr_url: str = ""):
    return {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[0],
        "height": image_size[1],
        "date_captured": date_captured,
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url
    }


def create_annotation_info(annotation_id: int, image_id: int, category_id: int, is_crowd: int,
                           area: float,
                           bounding_box: Tuple[float, float, float, float], segmentation):
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": is_crowd,
        "area": area,  # float
        "bbox": bounding_box,  # [x,y,width,height]
        "segmentation": segmentation  # [polygon]
    }


def convert(image_dir: str, annotation_path: str):
    """
    :param str image_dir: directory for your images
    :param str annotation_path: path for your annotations
    :return: coco_output is a dictionary of coco style which you could dump it into a json file
    as for keywords 'info','licenses','categories',you should modify them manually
    """

    # Get the name(type) and supercategory(super_type) from VIA ANNOTATION

    print('Start to reading')
    rows = []
    with open(annotation_path, newline='') as csvfile:
        annotation_reader = csv.DictReader(csvfile)
        for row in annotation_reader:
            rows.append(row)
    print(rows[0])

    print('Collect all categories')
    # Collect all categories
    category_names = set([])
    for row in rows:
        region_attributes = row['region_attributes']
        region_attributes = json.loads(region_attributes)
        if 'type' in region_attributes:
            category_names.update([region_attributes['type']])
    coco_categories = [{'id': i + 1, 'name': category, 'supercategory': ''} for i, category in
                       enumerate(category_names)]

    # Template for output
    coco_output = {
        'info': {
            "description": "Example Dataset",
            "url": "https://github.com/somal/VIA2COCO",
            "version": "0.1.0",
            "year": 2021,
            "contributor": "somal",
            "date_created": datetime.utcnow().isoformat(' ')},
        'licenses': [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "https://creativecommons.org/licenses/by-nc-sa/2.0/"
            }
        ],
        'categories': coco_categories,
        'images': [],
        'annotations': []
    }

    print('Parsing')
    annotation_id = 0
    for image_id, row in enumerate(rows):
        filename = row['filename']
        img = cv2.imread(join(image_dir, filename))
        if img is None:
            continue
        # make image info and storage it in coco_output['images']
        image_info = create_image_info(image_id, filename, img.shape[:2])
        coco_output['images'].append(image_info)

        region_attributes = row['region_attributes']
        region_attributes = json.loads(region_attributes)
        if not 'type' in region_attributes:
            continue
        category = region_attributes['type']
        # cate must in categories
        assert category in category_names
        # get the cate_id
        cate_id = 0
        for category in coco_output['categories']:
            if category == category['name']:
                cate_id = category['id']
        ####################################################################################################

        iscrowd = 0
        box_dict = row['region_shape_attributes']
        box_dict = json.loads(box_dict)

        x, y, w, h = box_dict['x'], box_dict['y'], box_dict['width'], box_dict['height']
        box = (x, y, w, h)
        area = w * h
        segmentation = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        ann_info = create_annotation_info(annotation_id, image_id, cate_id, iscrowd, area, box, segmentation)
        coco_output['annotations'].append(ann_info)
        annotation_id += 1
    print('Finish')

    return coco_output


if __name__ == '__main__':
    IMG_FOLDER_PATH = '../bd_presales/cable_line/datasets/21'
    ANNOTATIONS_FILE_PATH = join(IMG_FOLDER_PATH, 'lbl', 'via_annotation2.csv')

    # Convert VIA annotations to COCO annotations
    annotations = convert(image_dir=IMG_FOLDER_PATH, annotation_path=ANNOTATIONS_FILE_PATH)

    # Save COCO annotations
    with open(join(IMG_FOLDER_PATH, 'COCO_annotation.json'), 'w', encoding="utf-8") as outfile:
        json.dump(annotations, outfile, sort_keys=True, indent=4, ensure_ascii=False)
