from text_detection import Detector
from reader import Reader
import os
import cv2
import matplotlib.pyplot as plt
import time
from PIL import Image

# file config and weights
yolo_weight = os.path.join("./models", "yolo_tinyv4_passport_best.weights")
yolo_cfg = os.path.join("./models", "yolo_tinyv4_passport.cfg")
class_names = os.path.join("./models", "passport.txt")
transformer_weight = os.path.join("./models", "transformerocr_new.pth")

image_path = os.path.join("./passport", "passport", "1828477_ho_chieu_bui_thi_hong_duong_0.jpg")
#image_path = "./passport/20201006135813.jpg"

detector = Detector(yolo_weight, yolo_cfg, (416, 416), class_names)
reader = Reader(transformer_weight)

start = time.time()
# convert image to ndarray
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# text detection
classes, scores, boxes, inference_time = detector.inference(image)
fn_img, pob_img = detector.crop(image, classes, boxes)

# convert ndarray to PIL image
fn_img = Image.fromarray(fn_img)
pob_img = Image.fromarray(pob_img)

# text recognition
full_name = reader.read(fn_img)
place_of_birth = reader.read(pob_img)
place_of_birth = reader.postprocess_address(place_of_birth, 5)

print("Full name: ", full_name)
print("Place of birth: ", place_of_birth)
print("Total time: ", time.time() - start)
