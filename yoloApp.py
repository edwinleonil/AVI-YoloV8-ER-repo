import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import os
import numpy as np
from ultralytics import YOLO

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Viewer")
        self.master.geometry("660x710")

        self.photo = None

        # Load a pretrained YOLO model (recommended for training)
        self.model = YOLO('runs/detect/train/weights/best.pt')

        # specify the path to the folder containing the images
        self.folder_path = 'dataset/images/train'

        # get a list of all the files in the folder
        self.file_list = os.listdir(self.folder_path)

        # set the current image index to 0
        self.image_index = 0

        # create a canvas to display the image
        self.canvas = tk.Canvas(self.master, width=640, height=640)
        self.canvas.grid(row=0, column=0, padx=10, pady=10, sticky='we')

        # create a button to go to the next image
        self.next_button = tk.Button(self.master, text="Next Image", command=self.next_image)
        self.next_button.grid(row=1, column=0, padx=10, pady=10, sticky='we')

        # display the first image
        self.display_image()

    def display_image(self):
        # joint the folder path and the image name
        image_path = os.path.join(self.folder_path, self.file_list[self.image_index])

        # Run inference on a single image
        results = self.model(image_path)

        # get the label, confidence and xyxy from the results
        for r in results:
            label = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy().astype(float)
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)

            # joint the label, confidence and xyxy into a single list of one dimension
            label_conf_xyxy = np.concatenate((label.reshape(-1,1), conf.reshape(-1,1), xyxy), axis=1)

        # get p1 and p2
        p1 = label_conf_xyxy[:,2:4]
        p2 = label_conf_xyxy[:,4:6]

        # convert p1 and p2 to a list of tuples adn keep them as integers
        p1 = [tuple(map(int, p)) for p in p1]
        p2 = [tuple(map(int, p)) for p in p2]

        # open the image and convert it to a tkinter PhotoImage
        img = Image.open(image_path)
        img = img.convert('RGB')

        # draw the bounding boxes on the image
        draw = ImageDraw.Draw(img)
        for i in range(len(p1)):
            draw.rectangle((p1[i], p2[i]), outline=(124,255,0), width=2)

        # convert the image to a tkinter PhotoImage and add it to the canvas
        self.photo = ImageTk.PhotoImage(img)

        # create a white canvas the size of the image
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        
        # update the image on the canvas
        self.canvas.image = self.photo
        self.canvas.grid(row=0, column=0, padx=10, pady=10, sticky='we')
        
        # create a button to go to the next image
        self.next_button = tk.Button(self.master, text="Next Image", command=self.next_image)
        self.next_button.grid(row=1, column=0, padx=10, pady=10, sticky='we')

    def next_image(self):
        # increment the image index
        self.image_index += 1

        # if we've reached the end of the file list, start over
        if self.image_index == len(self.file_list):
            self.image_index = 0

        # clear the canvas
        self.canvas.delete("all")

        # display the next image
        self.display_image()

root = tk.Tk()
app = App(root)
root.mainloop()