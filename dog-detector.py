import torch
import numpy as np
import cv2
from time import time
from time import sleep
import datetime
import keyboard


class DogDetection:

    def __init__(self):

        self.model = self.load_model()
        self.image_directory = r'C:\Users\Ben\Documents\dog-detector\images'
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s6', pretrained=True)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        results = self.model([frame])
        labels, cord = results.xyxyn[0][:, -1].to('cpu').numpy(), results.xyxyn[0][:, :-1].to('cpu').numpy()
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        shouldISave = False
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                # if there's a dog or a person in the image, we will save the image
                if self.class_to_label(labels[i]) == "dog":
                    shouldISave = True
                if self.class_to_label(labels[i]) == "person":
                    shouldISave = True

        return frame, shouldISave

    def __call__(self):
        print("Running Dog Pooping Detection Algorithm!")
        player = cv2.VideoCapture(0)
        assert player.isOpened()
        player.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        player.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("width: "+str(x_shape)+" height: "+str(y_shape))
        while True:
            ret, frame = player.read()
            if not ret:
                break
            results = self.score_frame(frame)
            frame, shouldISave = self.plot_boxes(results, frame)
            # cv2.imshow('frame',frame) #if we want to view the webcam feed, uncomment this line
            if shouldISave:
                cv2.imwrite(self.image_directory + "\image"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+".jpg", frame)
                print("Saved image at "+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
            if cv2.waitKey(1) == ord('q'):
                print("About to quit. One moment. Cleaning up.")
                sleep(5)
                break
            if keyboard.is_pressed('q'):
                print("About to quit. One moment. Cleaning up.")
                sleep(5)
                break
            sleep(3)
        player.release()

a = DogDetection()
a()