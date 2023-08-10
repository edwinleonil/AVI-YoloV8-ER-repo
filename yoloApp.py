import os
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO

# specify the path to the folder containing the images
folder_path = 'dataset/images/train'

# get a list of all the files in the folder
file_list = os.listdir(folder_path)

# filter out only the image files
image_list = [file for file in file_list if file.endswith('.jpg') or file.endswith('.png')]

# create a YOLO model
model = YOLO('yolov8n.pt')

# create a Tkinter window
window = tk.Tk()

# set the window title
window.title('Image Viewer')

# set the window size
window.geometry('800x600')

# create a label to display the image
image_label = tk.Label(window)
image_label.pack()

# create a button to predict the next image
def predict_next_image():
    # get the next image in the list
    global current_image_index
    current_image_index += 1
    if current_image_index >= len(image_list):
        current_image_index = 0
    image_path = os.path.join(folder_path, image_list[current_image_index])
    
    # run inference on the image
    results = model(image_path)
    
    # display the image with bounding boxes and labels
    image = Image.open(image_path)
    image_with_boxes = results.render()
    photo = ImageTk.PhotoImage(image_with_boxes)
    image_label.config(image=photo)
    image_label.image = photo

# create a button to predict the next image
next_button = tk.Button(window, text='Predict Next', command=predict_next_image)
next_button.pack()

# initialize the current image index
current_image_index = 0

# predict the first image
predict_next_image()

# start the Tkinter event loop
window.mainloop()