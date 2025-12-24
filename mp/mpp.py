import cv2
import numpy as np
from gtts import gTTS
import pygame
import os
import time



# Ensure YOLO files are in place
if not os.path.exists("yolov4.weights"):
    print("Downloading yolov4.weights...")
    os.system("curl -L -o yolov4.weights https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights")

if not os.path.exists("yolov4.cfg"):
    print("Downloading yolov4.cfg...")
    os.system("curl -L -o yolov4.cfg https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg")

if not os.path.exists("coco.names"):
    print("Downloading coco.names...")
    os.system("curl -L -o coco.names https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names")

# Load YOLO
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize video capture for the webcam
video = cv2.VideoCapture(0)
labels = []

# Initialize Pygame for playing the TTS audio
pygame.mixer.init()

def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("tts.mp3")
    pygame.mixer.music.load("tts.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # Wait for the music to finish playing
        time.sleep(0.1)
    pygame.mixer.music.unload()  # Unload the file after it finishes playing
    os.remove("tts.mp3")  # Remove the file to avoid permission issues

# Record the start time
start_time = time.time()

# Run loop for at least 2 minutes
while True:
    ret, frame = video.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    detected_labels = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            detected_labels.append(label)
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    # Check for new labels to speak
    if labels != detected_labels:
        labels = detected_labels
        if labels:
            speak(" and ".join(labels))

    cv2.imshow("Image", frame)

    # Check if 2 minutes have passed
    if time.time() - start_time >= 120:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
pygame.quit()