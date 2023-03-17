import os
import sys
import math
import glob
import logging
import random
import cv2
import numpy as np
from typing import List


class MaskCutMix:
    def __init__(self, 
                 crop_obj_dir: str, 
                 crop_class_names: List[str], 
                 class_names: List[str],
                 cache_images=True):
        """_summary_

        :param crop_obj_dir: _description_
        :param crop_class_names: _description_
        :param class_names: _description_
        """
        
        self.crop_obj_dir = crop_obj_dir
        self.class_names = class_names
        self.crop_class_names = []
        self.cache_images = cache_images
        
        # self.logger = logging.getLogger(__name__)
        # self.logger.setLevel(logging.DEBUG)
        
        # Conformity of cls names and cls ids (cls_name_1 - 0, cls_name_2 - 1, ...)
        self.class_conformity = {cls_name: id for id, cls_name in enumerate(class_names)}
        #self.class_names_id = {class_name: class_id for class_id, class_name in enumerate(class_names)}

        self.images = {} # dict with keys - crop class name, values - list of images
        for crop_class_name in crop_class_names:
            crop_class_dir = os.path.join(self.crop_obj_dir, crop_class_name)
            
            # If crop class has no dir with images, then skip
            if not os.path.exists(crop_class_dir):
                self.images.update({crop_class_name: None})
                continue
            
            self.crop_class_names.append(crop_class_name)
            
            # Add list of image paths to common dict
            if cache_images:
                img_list = [cv2.imread(os.path.join(crop_class_dir, fname)) for fname in os.listdir(crop_class_dir)]
                self.images[crop_class_name] = img_list
            else:
                img_list = [os.path.join(crop_class_dir, fname) for fname in os.listdir(crop_class_dir)]
                self.images[crop_class_name] = img_list
            # self.logger.info("Generate object from crop images")
            # self.logger.info(f"{crop_class_name}: count images {len(img_paths_list)}")

    def __call__(self, im, labels, p=1.0, num_obj=10, **kwargs):
        height, width = im.shape[:2]
        
        if random.random() > p:
            return im, labels

        # random insert image coco objects in original image
        for idx in range(num_obj):
            class_coco_name = self.select_class_name()
            
            img_obj = self.select_image(class_coco_name)
            if img_obj is None:
                continue
            
            im, bbox = self.random_paste_img_to_img(im, img_obj, labels)
            if bbox is None:
                continue
        
            class_id = self.class_conformity.get(class_coco_name)
            x_c, y_c, w_bbox, h_bbox = self.xyxy_to_xcycwh(bbox)
            x_c, y_c = x_c / width, y_c / height
            w_bbox, h_bbox = w_bbox / width, h_bbox / height
            yolo_bbox = np.array([[class_id, x_c, y_c, w_bbox, h_bbox]])
            yolo_bbox = self.fix_bbox(yolo_bbox)
            labels = np.append(labels, yolo_bbox, axis=0)

        return im, labels
    
    def select_class_name(self) -> str:
        crop_class_number = random.randint(0, len(self.crop_class_names) - 1)
        selected_class_name = self.crop_class_names[crop_class_number]
        return selected_class_name
    
    def select_image(self, class_name: str) -> np.ndarray:
        
        if self.cache_images:
            img = random.choice(self.images[class_name])
        else:
            path_img = random.choice(self.images[class_name])
            img = cv2.imdecode(np.fromfile(path_img, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img
    
    def random_paste_img_to_img(self, img: np.ndarray, img_obj: np.ndarray, labels: np.ndarray):
        height, width = img.shape[0:2]
        h_obj, w_obj = img_obj.shape[0:2]

        x_tl = (width - w_obj) // 2
        x_br = (width + w_obj) // 2
        y_tl = (height - h_obj) // 2
        y_br = (height + h_obj) // 2
        bbox = [x_tl, y_tl, x_br, y_br]

        if w_obj > width or h_obj > height:
            return img, None
        
        new_bbox = [-1, -1, -1, -1]
        iou_max = 0
        for sample_id in range(10):
            warp_mat = self.get_random_perspective_transform(img.shape)

            # place img_obj on black background with size of original img
            img_obj_on_black = np.zeros(img.shape, dtype=img.dtype)
            img_obj_on_black[y_tl:y_br, x_tl:x_br] = img_obj

            # warp img obj and find mask of new obj position
            new_img_obj = cv2.warpPerspective(img_obj_on_black, warp_mat, (width, height))
            ret, mask = cv2.threshold(cv2.cvtColor(new_img_obj, cv2.COLOR_BGR2GRAY), 35, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            
            # find contour and its bbox of new obj position
            obj_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in obj_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                
                if new_bbox[0] == -1:
                    new_bbox = [x, y, x + w, y + h]
                    continue
                
                new_bbox[0] = min(new_bbox[0], x)
                new_bbox[1] = min(new_bbox[1], y)
                new_bbox[2] = max(new_bbox[2], x + w)
                new_bbox[3] = max(new_bbox[3], y + h)

            iou_max = self.get_iou_max(img, new_bbox, labels)
            if iou_max <= 0.1:
                break
            
            new_bbox = [-1, -1, -1, -1]
            
        if iou_max > 0.1:
            return img, None
        
        # if we cant find new bbox then skip it and return none
        if new_bbox == [-1, -1, -1, -1]:
            return img, None
        
        opacity = random.uniform(0.5, 1.0)

        img_without_obj = cv2.bitwise_and(img, img, mask=mask_inv)
        img_obj_placeholder = cv2.bitwise_and(img, img, mask=mask)
        
        final_img = cv2.addWeighted(img_obj_placeholder, 1 - opacity, new_img_obj, opacity, 0)
        final_img = cv2.addWeighted(final_img, 1, img_without_obj, 1, 0)
        
        return final_img, new_bbox
    
    def get_random_perspective_transform(self,
                                         img_shape: tuple,
                                         degrees=20,
                                         translate=0.4,
                                         scale=1.0,
                                         scale_range=0.1,
                                         shear=30,
                                         perspective=0.0) -> np.ndarray:
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
        cls_id, xc, yc, w, h = yolo_bbox[0]

        xc = min(1, max(0, xc))
        yc = min(1, max(0, yc))
        w = min(2 * xc, 2 - 2 * xc, w)
        h = min(2 * yc, 2 - 2 * yc, h)

        yolo_bbox[0][1] = xc
        yolo_bbox[0][2] = yc
        yolo_bbox[0][3] = w
        yolo_bbox[0][4] = h

        return yolo_bbox
    
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



def get_tube_img(img: np.ndarray) -> np.ndarray:
    
    gray = cv2.split(img)[2] #cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 4)
    ret, binary = cv2.threshold(gray, 75, 255, cv2.THRESH_TRIANGLE)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    max_area = 0
    max_id = -1
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > max_area:
            max_area = cv2.contourArea(contour)
            max_id = i
    
    rect = cv2.minAreaRect(contours[max_id])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    crop_box = np.array(
        [[0, 0],
         [0, img.shape[1]],
         [img.shape[0], img.shape[1]],
         [img.shape[0], 0]],
        dtype=np.int64        
    )
    warp_mat = cv2.getPerspectiveTransform(box.astype(np.float32), crop_box.astype(np.float32))
    crop_img = cv2.warpPerspective(img, warp_mat, img.shape[1::-1])
    
    return crop_img, warp_mat


if __name__ == '__main__':

    dataset_dir = r'D:\datasets\tmk\prepared\tmk_cvs1_yolo_640px_05032023'
    dir_coco_obj = r"D:\datasets\tmk\crops\0712_comet_crops"#"/home/student2/datasets/crops/0712_comet_crops"
    coco_class_names = ['comet']
    class_names = ['comet', 'joint', 'number']
    golf = MaskCutMix(dir_coco_obj, coco_class_names, class_names)
    
    splits = ['valid']
    for split in splits:
        images_dir = os.path.join(dataset_dir, split, 'images')
        labels_dir = os.path.join(dataset_dir, split, 'labels')

        img_files = os.listdir(images_dir)
        img_files.sort()
        for img_file in img_files:
            img_name = os.path.splitext(img_file)[0]
            print(split, img_name)
            img = cv2.imread(os.path.join(images_dir, img_file))
            if img is None:
                continue
            img = cv2.resize(img, (640, 535))
            
            labels = np.zeros((0, 5))
            new_img, new_labels = golf(img, labels)

            print(new_labels)
            
            for i in range(new_labels.shape[0]):
                cls_id, xc, yc, w, h = new_labels[i]
                x = int((xc - w/2) * img.shape[1])
                y = int((yc - h/2) * img.shape[0])
                w = int(w * img.shape[1])
                h = int(h * img.shape[0])
                cv2.rectangle(new_img,
                            (x, y),
                            (x + w, y + h),
                            (0, 255, 0), 5)
        
            cv2.imshow("test", new_img)
            if cv2.waitKey(0) == 27:
                break

