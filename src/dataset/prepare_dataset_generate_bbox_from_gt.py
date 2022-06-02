import os 
import glob
import json
import cv2
import multiprocessing
import time
import argparse

parser = argparse.ArgumentParser(description='Argument parser of Generate bbox.')
parser.add_argument('--thread_num', type=int, default=10, help='thread number')
# parser.add_argument('--pathLabel', type=str, default= '/dataset/TrafficLight_AIHUB/Validation/labels/', help='label of path')
# parser.add_argument('--pathImg', type=str, default= '/dataset/TrafficLight_AIHUB/Validation/images/', help='path of images')
# parser.add_argument('--pathBbox', type=str, default= '/dataset/TrafficLight_AIHUB/Validation/bbox/', help='save path of cropped bbox')

parser.add_argument('--pathImg', type=str, default= '/dataset/TrafficLight_AIHUB/Training/images_d_1/', help='path of images')
parser.add_argument('--pathLabel', type=str, default= '/dataset/TrafficLight_AIHUB/Training/labels_d_1/', help='label of path')

parser.add_argument('--pathBbox', type=str, default= '/dataset/TrafficLight_AIHUB/Training/bbox/', help='save path of cropped bbox')
args = parser.parse_args()

### Crop bbox from image
### input: 
###    - anno: light_count, box, attribute, type, class, direction
### output:
###    - save croped image corresponding classes.
###    - classes determined by anno->attibute
def image_crop_bbox(anno, imgName, _idx):
    _imgName = os.path.join(args.pathImg, imgName)
    imgSrc = cv2.imread(_imgName)
    box_X = anno['box'][0]
    box_Y = anno['box'][1]
    box_W = anno['box'][2]
    box_H = anno['box'][3]
    
    bboxImg = imgSrc[box_Y:box_H,box_X:box_W]
    #cv2.imshow(bboxImg)
    #cv2.waitKey(0)
    for _att in anno['attribute']:
        if _att['red'] == 'on':
            if _att['left_arrow'] == 'on':
                bboxPath = os.path.join(args.pathBbox,'red_left/' + str(_idx) + '_' + imgName)
            else:
                bboxPath = os.path.join(args.pathBbox,'red/' + str(_idx) + '_' + imgName)
        elif _att['green'] == 'on':
            if _att['left_arrow'] == 'on':
                bboxPath = os.path.join(args.pathBbox,'green_left/' + str(_idx) + '_' + imgName)
            else:
                bboxPath = os.path.join(args.pathBbox,'green/' + str(_idx) + '_' + imgName)
        elif _att['yellow'] == 'on':
            if _att['left_arrow'] == 'on':
                bboxPath = os.path.join(args.pathBbox,'yellow_left/' + str(_idx) + '_' + imgName)
            else:
                bboxPath = os.path.join(args.pathBbox,'yellow/' + str(_idx) + '_' + imgName)
        elif _att['left_arrow'] == 'on':
            bboxPath = os.path.join(args.pathBbox,'left/' + str(_idx) + '_' + imgName)
        else: 
            bboxPath = os.path.join(args.pathBbox,'off/' + str(_idx) + '_' + imgName)
    
    if box_H - box_Y > 10 and box_W - box_X > 10:
        cv2.imwrite(bboxPath,bboxImg)
    #print(imgName, box_X, box_W, box_Y, box_H)    
    
### Load each annotated file from json    
### perform:
###    - image_crop_bbox()
def load_annotation(label):
    with open(label) as json_file:
        label_data = json.load(json_file)
        _idx = 0
        for _annotation in label_data['annotation']:
            if _annotation['class'] == 'traffic_light':
                try:
                    if _annotation['type'] == 'car':
                        _idx += 1
                        image_crop_bbox(_annotation, label_data['image']['filename'], _idx)
                except:
                    pass
                    

if __name__=='__main__':
    labelPath = sorted(glob.glob(args.pathLabel + '*.json'))
    # for label in labelPath:
    #     load_annotation(label)
    pool = multiprocessing.Pool(processes=args.thread_num)
    pool.map(load_annotation, labelPath)
    pool.close()
    pool.join()
        