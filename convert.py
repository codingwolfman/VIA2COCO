import json
import os
from datetime import datetime
from typing import Tuple

import cv2

from get_area import get_area_of_polygon, x_y_to_points


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


def get_segmentation(coord_x, coord_y):
    seg = []
    for x, y in zip(coord_x, coord_y):
        seg.append(x)
        seg.append(y)
    return [seg]


def convert(image_dir: str, annotation_path: str):
    """
    :param str image_dir: directory for your images
    :param str annotation_path: path for your annotations
    :return: coco_output is a dictionary of coco style which you could dump it into a json file
    as for keywords 'info','licenses','categories',you should modify them manually
    """
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
        'categories': [
            {
                'id': 1,
                'name': 'rib',
                'supercategory': 'bone',
            },
            {
                'id': 2,
                'name': 'clavicle',
                'supercategory': 'bone',
            }
        ],
        'images': [],
        'annotations': []
    }

    ann = json.load(open(annotation_path))
    # annotations id start from zero
    ann_id = 0
    # in VIA annotations, keys are image name
    for img_id, key in enumerate(ann.keys()):

        filename = ann[key]['filename']
        img = cv2.imread(image_dir + filename)
        # make image info and storage it in coco_output['images']
        image_info = create_image_info(img_id, os.path.basename(filename), img.shape[:2])
        coco_output['images'].append(image_info)
        regions = ann[key]["regions"]
        # for one image ,there are many regions,they share the same img id
        for region in regions:
            cat = region['region_attributes']['label']
            assert cat in ['rib', 'clavicle']
            if cat == 'rib':
                cat_id = 1
            else:
                cat_id = 2
            iscrowd = 0
            points_x = region['shape_attributes']['all_points_x']
            points_y = region['shape_attributes']['all_points_y']

            area = get_area_of_polygon(x_y_to_points(points_x, points_y))
            min_x = min(points_x)
            max_x = max(points_x)
            min_y = min(points_y)
            max_y = max(points_y)
            box = (min_x, min_y, max_x - min_x, max_y - min_y)
            segmentation = get_segmentation(points_x, points_y)
            # make annotations info and storage it in coco_output['annotations']
            ann_info = create_annotation_info(ann_id, img_id, cat_id, iscrowd, area, box, segmentation)
            coco_output['annotations'].append(ann_info)
            ann_id = ann_id + 1

    return coco_output


if __name__ == '__main__':
    pass