import numpy as np
import cv2
import os
from .mask_cutmix import MaskCutMix
from .mask_cutmix import get_tube_img


class TubeMaskCutMix(MaskCutMix):
    
    def __call__(self, im, labels, p=1.0, num_obj=10, **kwargs):
        tube_img, warp_mat = get_tube_img(im)
        empty_labels = np.zeros((0, 5))
        tube_img, inserted_labels = super(TubeMaskCutMix, self).__call__(tube_img, empty_labels, p, num_obj, **kwargs)
        
        ret, inverse_warp_mat = cv2.invert(warp_mat)
        tube_img = cv2.warpPerspective(tube_img, inverse_warp_mat, (640, 640))
        
        ret, mask = cv2.threshold(cv2.cvtColor(tube_img, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY_INV)
        im = cv2.bitwise_and(im, im, mask=mask)
        im = cv2.add(im, tube_img)
        
        height, width = im.shape[:2]
        new_inserted_labels = []
        for i in range(inserted_labels.shape[0]):
            cls, xc, yc, w, h = inserted_labels[i]
            bbox = [width * (xc - w / 2),
                    height * (yc - h / 2),
                    width * (xc + w / 2),
                    height * (yc + h / 2)]
            new_bbox = self.warp_bbox(bbox, inverse_warp_mat, (height, width))

            x1, y1, x2, y2 = new_bbox
            xc = (x2 + x1) / 2 / width
            yc = (y2 + y1) / 2 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height

            new_inserted_labels.append([cls, xc, yc, w, h])
        
        labels = np.concatenate([labels, np.array(new_inserted_labels)], axis=0)
        
        return im, labels
    

if __name__ == '__main__':

    dataset_dir = '/home/student2/datasets/prepared/tmk_cvs3_yolo_640px_18032023'
    dir_coco_obj = '/home/student2/datasets/crops/1903_cvs3_defects_crops'
    coco_class_names = ['riska', 'sink']
    class_names = ['riska', 'sink']
    golf = TubeMaskCutMix(dir_coco_obj, coco_class_names, class_names)
    
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
            img = cv2.resize(img, (640, 640))
            
            labels = np.zeros((0, 5))
            new_img, new_labels = golf(img, labels)

            print(new_labels)
            
            for i in range(new_labels.shape[0]):
                cls_id, xc, yc, w, h = new_labels[i]
                x = int((xc - w/2) * img.shape[1])
                y = int((yc - h/2) * img.shape[0])
                w = int(w * img.shape[1])
                h = int(h * img.shape[0])
                cv2.putText(new_img, class_names[int(cls_id)], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                cv2.rectangle(new_img,
                            (x, y),
                            (x + w, y + h),
                            (0, 255, 0), 2)
        
            cv2.imshow("test", new_img)
            if cv2.waitKey(0) == 27:
                break



