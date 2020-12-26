import cv2
import numpy as np
import time
import os
import glob2


class Detector:
    CONFIDENCE_THRESHOLD = 0.2
    NMS_THRESHOLD = 0.5
    COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

    def __init__(self, yolotiny_weight, yolotiny_config, input_size, class_names):

        self.set_classes(class_names)
        self.input_size = input_size
        self.net = cv2.dnn.readNet(yolotiny_weight, yolotiny_config)
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=input_size, scale=1 / 255)

    def inference(self, image):
        start = time.time()
        classes, scores, boxes = self.model.detect(image, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
        if len(scores) < 2:
            raise Exception("Invalid Image")
        if scores.shape[0] > 2:

            max_args = np.argsort(scores.reshape(scores.shape[0], ))[:2]
            classes = classes[max_args]
            scores = scores[max_args]
            boxes = boxes[max_args]

            if scores[0][0] == scores[0][1]:
                raise  Exception("Invalid Image")

        end = time.time()
        inference_time = end - start

        return classes, scores, boxes, inference_time

    def draw_bbox(self, image, classes, scores, boxes):
        for (classid, score, box) in zip(classes, scores, boxes):
            color = self.COLORS[int(classid) % len(self.COLORS)]
            label = "%s : %f" % (self.classes[classid[0]], score)
            cv2.rectangle(image, box, color, 1)
            cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return image

    def crop(self, image, classes, boxes):
        full_name = []
        place_of_birth = []
        for cls in classes.flatten():
            if cls == 0:
                coordinate = boxes[cls]
                x, y, w, h = coordinate
                cropped_image = image[y:y+h, x:x + w, :]
                full_name.append(cropped_image)
                continue

            elif cls == 1:
                coordinate = boxes[cls]
                x, y, w, h = coordinate
                cropped_image = image[y: y+h, x: x + w, :]
                place_of_birth.append(cropped_image)

        return full_name[0], place_of_birth[0]

    def set_classes(self, class_names):
        with open(class_names, 'r') as f:
            self.classes = [cname.strip() for cname in f.readlines()]

    def get_classes(self):
        return self.classes


if __name__ == '__main__':
    yolotiny_weight = os.path.join("./models", "yolo_tinyv4_passport_best.weights")
    yolotiny_cfg = os.path.join("./models", "yolo_tinyv4_passport.cfg")
    class_names = os.path.join("./models", "passport.txt")

    detector = Detector(yolotiny_weight, yolotiny_cfg, (416, 416), class_names)

    image_path = os.path.join("./passport", "passport", "1f3789021102d0e0669a0e1c467b1f80.jpg")

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    classes, scores, boxes, inference_time = detector.inference(image)
    drawn_img = detector.draw_bbox(image, classes, scores, boxes)
    cv2.imshow('drawn_image', cv2.cvtColor(drawn_img, cv2.COLOR_RGB2BGR))
    cv2.waitKey()


