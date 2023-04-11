import os
import sys
import math
import glob
import logging
import random
import cv2
import numpy as np
from typing import List, Tuple


class TubeGenerator:
    def __init__(self, 
                 tube_images_dir: str,
                 tube_masks_dir: str,
                 tube_labels_dir: str,
                 crop_images_dir: str, 
                 crop_class_names: List[str], 
                 class_names: List[str]):
        
        self.tube_images_dir = tube_images_dir
        self.tube_masks_dir = tube_masks_dir
        self.tube_labels_dir = tube_labels_dir

        self.crop_images_dir = crop_images_dir
        self.class_names = class_names
        self.crop_class_names = []
        
        # Conformity of cls names and cls ids (cls_name_1 - 0, cls_name_2 - 1, ...)
        self.class_conformity = {cls_name: id for id, cls_name in enumerate(class_names)}
        
        self.crop_img_paths = {} # dict with keys - crop class name, values - list of images
        for crop_class_name in crop_class_names:
            crop_class_dir = os.path.join(self.crop_images_dir, crop_class_name)
            
            # If crop class has no dir with images, then skip
            if not os.path.exists(crop_class_dir):
                self.crop_img_paths.update({crop_class_name: None})
                continue
            
            self.crop_class_names.append(crop_class_name)
            
            # Add list of image paths to common dict
            img_list = [os.path.join(crop_class_dir, fname) for fname in os.listdir(crop_class_dir)]
            self.crop_img_paths[crop_class_name] = img_list
        
        self.tube_img_paths = [os.path.join(tube_images_dir, fname) for fname in os.listdir(tube_images_dir)]
    
    def __call__(self, p=1.0, num_paste=5, **kwargs):
        
        tube_img, tube_mask, labels = self.select_tube_data()
        tube_h = self.get_tube_homography(tube_mask)
        height, width = tube_img.shape[:2]
        
        if random.random() > p:
            return tube_img, labels

        for idx in range(num_paste):
            selected_class_name = self.select_class_name()
            crop_img = self.select_crop_image(selected_class_name)

            if crop_img is None:
                continue
            
            tube_img, bbox = self.random_paste_crop_to_obj(tube_img, crop_img, labels, tube_h)
            if bbox is None:
                continue
        
            class_id = self.class_conformity.get(selected_class_name)
            x_c, y_c, w_bbox, h_bbox = self.xyxy_to_xcycwh(bbox)
            x_c, y_c = x_c / width, y_c / height
            w_bbox, h_bbox = w_bbox / width, h_bbox / height

            yolo_bbox = [class_id, x_c, y_c, w_bbox, h_bbox]
            yolo_bbox = self.fix_bbox(yolo_bbox)
            labels.append(yolo_bbox)

        labels = np.array(labels) if len(labels) != 0 else np.zeros((0, 5))

        return tube_img, labels
    
    def select_tube_data(self) -> Tuple[np.ndarray, np.ndarray, List[List[float]]]:
        img_path = random.choice(self.tube_img_paths)
        filename = os.path.split(img_path)[-1]
        name, ext = os.path.splitext(filename)

        mask_path = os.path.join(self.tube_masks_dir, name + ext)
        lbl_path = os.path.join(self.tube_labels_dir, name + '.txt')
        
        img = cv2.imdecode(np.fromfile(img_path, dtype='uint8'), cv2.IMREAD_COLOR)

        if os.path.exists(mask_path):
            mask = cv2.imdecode(np.fromfile(mask_path, dtype='uint8'), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.ones(img.shape[:2], dtype='uint8') * 255
        
        if os.path.exists(lbl_path):
            labels = self.read_yolo_labels(lbl_path)
        else:
            labels = []

        return img, mask, labels
        
    
    def select_class_name(self) -> str:
        crop_class_number = random.randint(0, len(self.crop_class_names) - 1)
        selected_class_name = self.crop_class_names[crop_class_number]
        return selected_class_name
    
    def select_crop_image(self, class_name: str) -> np.ndarray:
        img_path = random.choice(self.crop_img_paths[class_name])
        img = np.load(img_path)
        return img
    
    def get_tube_homography(self, mask: np.ndarray) -> np.ndarray:
        
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if len(contours) == 0:
            return np.eye(4)
        
        max_area = 0
        max_id = -1
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > max_area:
                max_area = cv2.contourArea(contour)
                max_id = i
        
        rect = cv2.minAreaRect(contours[max_id])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        img_borders = np.array(
            [[0, 0],
            [0, mask.shape[1]],
            [mask.shape[0], mask.shape[1]],
            [mask.shape[0], 0]],
            dtype=np.int64        
        )
        warp_mat = cv2.getPerspectiveTransform(img_borders.astype(np.float32), box.astype(np.float32))
        
        return warp_mat
    
    def random_paste_crop_to_obj(self, obj_img: np.ndarray, crop_img: np.ndarray, labels: np.ndarray, obj_warp_mat: np.ndarray):
        height, width = obj_img.shape[0:2]
        h, w = crop_img.shape[0:2]

        x_tl = (width - w) // 2
        x_br = (width + w) // 2
        y_tl = (height - h) // 2
        y_br = (height + h) // 2
        bbox = [x_tl, y_tl, x_br, y_br]

        if w > width or h > height:
            return obj_img, None
        
        new_bbox = None
        iou_max = 0
        for sample_id in range(10):
            warp_mat = self.get_random_perspective_transform(obj_img.shape)

            # place img_obj on black background with size of original img
            crop_on_black = np.zeros((*obj_img.shape[1::-1], 4), dtype=crop_img.dtype)
            crop_on_black[y_tl:y_br, x_tl:x_br] = crop_img

            # warp img obj and find mask of new obj position
            new_crop_img = cv2.warpPerspective(crop_on_black, obj_warp_mat @ warp_mat, (width, height))
            new_crop_bgr = new_crop_img[:, :, 0:3]
            mask = new_crop_img[:, :, 3].astype('uint8')
            # mask_inv = cv2.bitwise_not(mask)
            
            # find contour and its bbox of new obj position
            obj_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(obj_contours) == 0:
                continue

            new_bbox = None
            for cnt in obj_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                
                if new_bbox is None:
                    new_bbox = [x, y, x + w, y + h]
                    continue
                
                new_bbox[0] = min(new_bbox[0], x)
                new_bbox[1] = min(new_bbox[1], y)
                new_bbox[2] = max(new_bbox[2], x + w)
                new_bbox[3] = max(new_bbox[3], y + h)

            iou_max = self.get_iou_max(obj_img, new_bbox, labels)
            if iou_max <= 0.1:
                break
            
            
        if iou_max > 0.1:
            return obj_img, None
        
        # if we cant find new bbox then skip it and return none
        if new_bbox is None:
            return obj_img, None
        
        final_img = obj_img.astype('float32')
        new_crop_bgr = new_crop_bgr.astype('float32') * 2
        #bgr_new_crop_img = cv2.normalize(bgr_new_crop_img, bgr_new_crop_img, 100, -100, cv2.NORM_MINMAX)
        
        final_img = final_img + new_crop_bgr
        final_img = np.clip(final_img, 0, 255)
        final_img = final_img.astype('uint8')
        
        return final_img, new_bbox
    
    def get_random_perspective_transform(self,
                                         img_shape: tuple,
                                         degrees=50,
                                         translate=0.4,
                                         scale=0.5,
                                         scale_range=0.2,
                                         shear=5,
                                         perspective=0.0001) -> np.ndarray:
        height = img_shape[0]
        width = img_shape[1]

        # Center
        C = np.eye(3)
        C[0, 2] = -img_shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img_shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(scale - scale_range, scale + scale_range)
        # s = random.uniform(1 - scale, 1 + scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT

        return M

    def warp_bbox(self, bbox: list, warp_mat: np.ndarray, img_size: tuple) -> list:
        x1, y1, x2, y2 = bbox
        corners = np.single([[[x1, y1],
                              [x1, y2],
                              [x2, y1],
                              [x2, y2]]])
        new_corners = cv2.perspectiveTransform(corners, warp_mat)

        new_x1 = int(new_corners[0, :, 0].min())
        new_x2 = int(new_corners[0, :, 0].max())
        new_y1 = int(new_corners[0, :, 1].min())
        new_y2 = int(new_corners[0, :, 1].max())
        
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(img_size[1] - 1, new_x2)
        new_y2 = min(img_size[0] - 1, new_y2)

        new_bbox = [new_x1, new_y1, new_x2, new_y2]
        return new_bbox
    
    def fix_bbox(self, yolo_bbox):
        cls_id, xc, yc, w, h = yolo_bbox

        xc = min(1, max(0, xc))
        yc = min(1, max(0, yc))
        w = min(2 * xc, 2 - 2 * xc, w)
        h = min(2 * yc, 2 - 2 * yc, h)

        yolo_bbox[1] = xc
        yolo_bbox[2] = yc
        yolo_bbox[3] = w
        yolo_bbox[4] = h

        return yolo_bbox
    
    def read_yolo_labels(self, path: str) -> List[List[float]]:
        labels = []
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        lines = text.split('\n')
        for line in lines:
            if line == '':
                continue

            line = line.split(' ')
            line = list(map(float, line))
            labels.append(line)
        
        return labels
        

    

    def xyxy_to_xywh(self, bbox):
        x_tl, y_tl, x_br, y_br = bbox
        return x_tl, y_tl, x_br - x_tl, y_br - y_tl

    def xyxy_to_xcycwh(self, bbox):
        x_tl, y_tl, x_br, y_br = bbox
        w, h = x_br - x_tl, y_br - y_tl
        return x_tl + w//2, y_tl + h//2, w, h

    def xywh_to_xcycwh(self, bbox):
        x_tl, y_tl, w, h = bbox
        return x_tl + w//2, y_tl + h//2, w, h
        
    def xcycwh_to_xyxy(self, bbox):
        x_c, y_c, w, h = bbox
        return x_c - w//2, y_c - h//2, x_c + w//2, y_c + h//2
    
    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    
    def get_iou_max(self, img, bboxA, labels_yolo):
        """
        Get maximum IoU bbox object in labels_yolo
        :param img: original image
        :param bboxA: bbox object
        :param labels_yolo: all coordinates bboxes in image
                           (format yolo - class_id, x_c, y_c, w, h)
        """
        height, width = img.shape[:2]
        iou_max = 0
        for bbox_yolo in labels_yolo:
            bboxB = bbox_yolo[1:].copy()
            b_xc, b_yc, b_w, b_h = bboxB
            b_xc, b_w = b_xc * width, b_w * width
            b_yc, b_h = b_yc * height, b_h * height
            bboxB = self.xcycwh_to_xyxy([b_xc, b_yc, b_w, b_h])
            iou = self.bb_intersection_over_union(bboxA, bboxB)
            if iou > iou_max:
                iou_max = iou
        return iou_max


def draw_bboxes(img, xcycwhn_bboxes):
    height, width = img.shape[:2]
    for bbox in xcycwhn_bboxes:
        cls_id, xc, yc, w, h = bbox
        xc = int(xc * width)
        yc = int(yc * height)
        w = int(w * width)
        h = int(h * height)
        x = xc - w // 2
        y = yc - h // 2
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)


def test_cvs1():
    tube_images_dir = '/mnt/data/tmk_datasets/other/tmk_cvs1_yolo_640px_14032023_tubes/train/nps'
    tube_labels_dir = '/mnt/data/tmk_datasets/other/tmk_cvs1_yolo_640px_14032023_tubes/train/labels'
    crop_images_dir = '/mnt/data/tmk_datasets/crops/0404_defect_crops' 
    crop_class_names = ['comet'] 
    class_names = ['comet', 'joint', 'number']

    gen = TubeGenerator(tube_images_dir, tube_labels_dir, crop_images_dir, crop_class_names, class_names)

    while True:
        img, labels = gen()
        cv2.imshow('img', img)
        draw_bboxes(img, labels)
        print(labels)
        cv2.imshow('img_labels', img)
        if cv2.waitKey() == 27:
            break
        
def test_cvs3():
    tube_images_dir = '/mnt/data/tmk_datasets/prepared/tmk_cvs3_yolo_640px_18032023/train/images'
    tube_masks_dir =  '/mnt/data/tmk_datasets/prepared/tmk_cvs3_yolo_640px_18032023/train/masks'
    tube_labels_dir = '/mnt/data/tmk_datasets/prepared/tmk_cvs3_yolo_640px_18032023/train/labels'
    crop_images_dir = '/mnt/data/tmk_datasets/crops/1004_defect_crops' 
    
    crop_class_names = ['sink', 'riska'] 
    class_names = ['other', 'tube', 'sink', 'riska', 'pseudo']

    gen = TubeGenerator(tube_images_dir, tube_masks_dir, tube_labels_dir, 
                        crop_images_dir, crop_class_names, class_names)

    while True:
        img, labels = gen()
        cv2.imshow('img', img)
        draw_bboxes(img, labels)
        print(labels)
        cv2.imshow('img_labels', img)
        if cv2.waitKey() == 27:
            break
        

if __name__ == '__main__':
    test_cvs3()

    
