from ultralytics import YOLO
import os
import cv2

# Load a pretrained YOLO model (recommended for training)
model = YOLO('runs/detect/train/weights/best.pt')

# specify the path to the folder containing the images
folder_path = 'dataset/images/train/'

# get a list of all the files in the folder
file_list = os.listdir(folder_path)

# PRINT THE FIRST 10 IMAGES
image = file_list[8]
file_path = 'dataset/images/train/Part 7_A00756786_Res_3_Normals X.png'

# Run inference on a single image
# results = model('dataset/images/train/'+image, save=True)

# Use the model to detect object - goat
result = model.predict(source=file_path, save=False, show=True)


# keep the window open until we press a key
cv2.waitKey(0)

# close all windows
cv2.destroyAllWindows()


### we can use the trained model to label pinsites