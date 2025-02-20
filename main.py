import tkinter.messagebox
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import simpledialog, filedialog
from tkinter import YES, BOTH, BOTTOM, X, W, E
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from typing import List, Dict

from PIL import Image, ImageDraw
import PIL
import cv2 as cv
import numpy as np
import pickle
import os.path
import json

# Save the model details to a file
def save_model_details(proj_name, data, classes, clf):
    data = {
        'proj_name': proj_name,
        'data': data,
        'classes': classes,
        'clf': clf
    }
    try:
        with open(f"{proj_name}/{proj_name}_data.pickle", "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        raise e

def load_model_details(proj_name):
    try:
        if os.path.exists(f"{proj_name}/{proj_name}_data.pickle"):
            with open(f"{proj_name}/{proj_name}_data.pickle", "rb") as f:
                data = pickle.load(f)
            
            # Ensure all required keys are present in the loaded data
            if 'proj_name' not in data:
                data['proj_name'] = proj_name
            if 'data' not in data:
                data['data'] = None
            if 'classes' not in data:
                data['classes'] = []
            if 'clf' not in data:
                data['clf'] = LinearSVC()  # Default to LinearSVC if classifier not saved

            return data
    except Exception as e:
        raise e

class DrawingClassifier:

    def __init__(self):
        self.class1, self.class2, self.class3 = None, None, None
        self.class1_counter, self.class2_counter, self.class3_counter = None, None, None
        self.clf = None
        self.proj_name = None
        self.root = None
        self.image1 = None

        self.status_label = None
        self.canvas = None
        self.draw = None

        self.brush_width = 15
        
        self.classes = []
        self.data = None
        
        self.classes_prompt()
        self.init_gui()

    def classes_prompt(self):
        msg = Tk()
        msg.withdraw()

        self.proj_name = simpledialog.askstring("Project Name", "Please enter your project name down below!", parent=msg)
        if os.path.exists(self.proj_name):
            data = load_model_details(self.proj_name)
            if data:
                self.data = data['data']
                self.classes = data['classes']
                self.clf = data['clf']
                self.proj_name = data['proj_name']
        else:
            try:
                num_classes = simpledialog.askstring("Number of classes", f"How many shapes are you categorizing?", parent=msg)
            except ValueError as e:
                tkinter.messagebox.showinfo("Drawing Classifier", "Value must be a number!", parent=self.root)
            
            if num_classes:
                self.num_class = int(num_classes)
                for i in range(1, self.num_class + 1):
                    nth = "st" if i == 1 else "nd" if i == 2 else "rd" if i == 3 else "th"
                    self.classes.append({"name": simpledialog.askstring(f"Class {i}", f"What is the {i}{nth} class called?", parent=msg), "count": i, "img_counter": 1})
                
            self.clf = LinearSVC()
            os.mkdir(self.proj_name)
            os.chdir(self.proj_name)

            for i in range(len(self.classes)):
                os.mkdir(self.classes[i].get("name"))
                
            os.chdir("..")

    def init_gui(self):
        WIDTH = 500
        HEIGHT = 500
        WHITE = (255, 255, 255)

        self.root = tk.Tk()
        self.root.title(f"Drawing Classifier Alpha v0.2 - {self.proj_name}")

        # Canvas for drawing doodles and classification of doodle
        self.canvas = Canvas(self.root, width=WIDTH-10, height=HEIGHT-10, bg="white")
        self.canvas.pack(expand=YES, fill=BOTH)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image1 = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
        self.draw = ImageDraw.Draw(self.image1)

        btn_frame = ttk.Frame(self.root, padding="10 10 10 10")
        btn_frame.pack(fill=X, side=BOTTOM)

        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.columnconfigure(2, weight=1)
        buttons = []

        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12), padding=5)

        for i, dir in enumerate(self.classes):
            buttons.append(ttk.Button(btn_frame, text=dir.get('name'), command=lambda i=i: self.save(i + 1)))

        for i, button in enumerate(buttons):
            button.grid(row=0, column=i, sticky=W+E, padx=5, pady=5)

        bm_btn = ttk.Button(btn_frame, text="Brush-", command=self.brushminus)
        bm_btn.grid(row=1, column=0, sticky=W+E, padx=5, pady=5)

        clear_btn = ttk.Button(btn_frame, text="Clear", command=self.clear)
        clear_btn.grid(row=1, column=1, sticky=W+E, padx=5, pady=5)

        bp_btn = ttk.Button(btn_frame, text="Brush+", command=self.brushplus)
        bp_btn.grid(row=1, column=2, sticky=W+E, padx=5, pady=5)

        train_btn = ttk.Button(btn_frame, text="Train Model", command=self.train_model)
        train_btn.grid(row=2, column=0, sticky=W+E, padx=5, pady=5)

        save_btn = ttk.Button(btn_frame, text="Save Model", command=self.save_model)
        save_btn.grid(row=2, column=1, sticky=W+E, padx=5, pady=5)

        load_btn = ttk.Button(btn_frame, text="Load Model", command=self.load_model)
        load_btn.grid(row=2, column=2, sticky=W+E, padx=5, pady=5)

        change_btn = ttk.Button(btn_frame, text="Change Model", command=self.rotate_model)
        change_btn.grid(row=3, column=0, sticky=W+E, padx=5, pady=5)

        predict_btn = ttk.Button(btn_frame, text="Predict", command=self.predict)
        predict_btn.grid(row=3, column=1, sticky=W+E, padx=5, pady=5)

        save_everything_btn = ttk.Button(btn_frame, text="Save Everything", command=self.save_everything)
        save_everything_btn.grid(row=3, column=2, sticky=W+E, padx=5, pady=5)

        self.status_label = ttk.Label(btn_frame, text=f"Current Model: {type(self.clf).__name__}")
        self.status_label.config(font=("Arial", 10))
        self.status_label.grid(row=4, column=1, sticky=W+E, padx=5, pady=5)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.attributes("-topmost", True)
        self.root.mainloop()

    def paint(self, event):
        try:
            x1, y1 = (event.x - 1), (event.y - 1)
            x2, y2 = (event.x + 1), (event.y + 1)
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", width=self.brush_width)
            self.draw.rectangle([x1, y2, x2 + self.brush_width, y2 + self.brush_width], fill="black", width=self.brush_width)
        except Exception as e:
            tkinter.messagebox.showerror("Error", str(e), parent=self.root)

    def save(self, class_num):
        try:
            self.image1.save("temp.png")
            img = PIL.Image.open("temp.png")
            img.thumbnail((50, 50), PIL.Image.LANCZOS)
            
            # Loop through classes list and use names to save in dirs
            for i, class_name in enumerate(self.classes):
                if class_num == class_name.get('count'):
                    img.save(f"{self.proj_name}/{class_name.get('name')}/{class_name.get('img_counter')}.png", "PNG")
                    class_name["img_counter"] += 1
                    
            save_model_details(self.proj_name, self.data, self.classes, self.clf)
            self.clear()
        except Exception as e:
            tkinter.Message.showerror("Error", str(e), parent=self.root)

    def brushminus(self):
        if self.brush_width <= 1:
            tkinter.messagebox.showwarning("Warning", f"Brush width is {self.brush_width}", parent=self.root)
        self.brush_width -= 1

    def brushplus(self):
        if self.brush_width >= 20:
            tkinter.messagebox.showwarning("Warning", f"Brush width is {self.brush_width}", parent=self.root)
        self.brush_width += 1

    def clear(self):
        try:
            self.canvas.delete("all")
            self.draw.rectangle([0, 0, 1000, 1000], fill="white")
        except Exception as e:
            tkinter.messagebox.showerror("Error", str(e), parent=self.root)

    def train_model(self):
        try:
            img_list = []
            class_list = []
            
            for i, class_info in enumerate(self.classes):
                class_label = i + 1  # Class labels start from 1
                class_name = class_info.get('name')
                img_counter = class_info.get('img_counter')
                
                for j in range(1, img_counter):
                    img_path = f"{self.proj_name}/{class_name}/{j}.png"
                    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = img.reshape(2500)
                        img_list.append(img)
                        class_list.append(class_label)
                    else:
                        print(f"Warning: Image at {img_path} could not be loaded.")
            
            if len(img_list) == 0 or len(class_list) == 0:
                tkinter.messagebox.showerror("Training Error", "No images found for training. Ensure that images are saved correctly.", parent=self.root)
                return

            img_list = np.array(img_list)
            class_list = np.array(class_list)
            
            self.clf.fit(img_list, class_list)
            tkinter.messagebox.showinfo("Drawing Classifier", "Model successfully trained!", parent=self.root)
        except Exception as e:
            tkinter.messagebox.showerror("Error", str(e), parent=self.root)

    def predict(self):
        try:
            self.image1.save("temp.png")
            img = PIL.Image.open("temp.png")
            img.thumbnail((50, 50), PIL.Image.LANCZOS)
            img = np.array(img)[:, :, 0]
            img = img.reshape(1, -1)
            class_num = self.clf.predict(img)[0]

            for class_info in self.classes:
                if class_info.get('count') == class_num:
                    tkinter.messagebox.showinfo("Drawing Classifier", f"The drawing is a {class_info.get('name')}", parent=self.root)

            self.clear()
        except Exception as e:
            tkinter.messagebox.showerror("Model Load Error", f"Model did not load successfully. Please train the model and try again.", parent=self.root)

    def rotate_model(self):
        try:
            if isinstance(self.clf, LinearSVC):
                self.clf = KNeighborsClassifier()
            elif isinstance(self.clf, KNeighborsClassifier):
                self.clf = LogisticRegression()
            elif isinstance(self.clf, LogisticRegression):
                self.clf = DecisionTreeClassifier()
            elif isinstance(self.clf, DecisionTreeClassifier):
                self.clf = RandomForestClassifier()
            elif isinstance(self.clf, RandomForestClassifier):
                self.clf = GaussianNB()
            elif isinstance(self.clf, GaussianNB):
                self.clf = LinearSVC()

            self.status_label.config(text=f"Current Model: {type(self.clf).__name__}")
        except Exception as e:
            tkinter.messagebox.showerror("Error", str(e), parent=self.root)

    def save_model(self):
        try:
            with open(f"{self.proj_name}/{self.proj_name}_model.pickle", "wb") as f:
                pickle.dump(self.clf, f)
            tkinter.messagebox.showinfo("Drawing Classifier", "Model successfully saved!", parent=self.root)
        except Exception as e:
            tkinter.messagebox.showerror("Error", str(e), parent=self.root)

    def load_model(self):
        try:
            if os.path.exists(f"{self.proj_name}/{self.proj_name}_model.pickle"):
                with open(f"{self.proj_name}/{self.proj_name}_model.pickle", "rb") as f:
                    self.clf = pickle.load(f)
                tkinter.messagebox.showinfo("Drawing Classifier", "Model successfully loaded!", parent=self.root)
                self.status_label.config(text=f"Current Model: {type(self.clf).__name__}")
        except Exception as e:
            tkinter.messagebox.showerror("Error", str(e), parent=self.root)

    def save_everything(self):
        try:
            save_model_details(self.proj_name, self.data, self.classes, self.clf)
            tkinter.messagebox.showinfo("Drawing Classifier", "Everything successfully saved!", parent=self.root)
        except Exception as e:
            tkinter.messagebox.showerror("Error", str(e), parent=self.root)

    def on_closing(self):
        self.save_everything()
        self.root.destroy()
        exit()
    
def load_model_details(proj_name):
    if os.path.exists(f"{proj_name}/{proj_name}_data.pickle"):
        try:
            with open(f"{proj_name}/{proj_name}_data.pickle", "rb") as f:
                data = pickle.load(f)
            
            # Ensure all required keys are present in the loaded data
            if 'proj_name' not in data:
                data['proj_name'] = proj_name
            if 'data' not in data:
                data['data'] = None
            if 'classes' not in data:
                data['classes'] = []
            if 'clf' not in data:
                data['clf'] = LinearSVC()  # Default to LinearSVC if classifier not saved

            return data
        except EXception as e:
            return e
    return None

def predict(image: Image, classifier: LinearSVC, classes: list) -> object:
    class_name = None
    try:
        # Load and preprocess the image
        img = PIL.Image.open(image)
        img.thumbnail((50, 50), PIL.Image.LANCZOS)
        img = np.array(img)[:, :, 0]
        img = img.reshape(1, -1)
    except FileNotFoundError as e:
        return f"File not found: {e}"
    except PIL.UnidentifiedImageError as e:
        return f"Cannot identify image file: {e}"
    except Exception as e:
        return f"Error processing image: {e}"
    
    try:
        # Predict the class
        class_num = classifier.predict(img)[0]
    except ValueError as e:
        return f"Error in model prediction: {e}"
    except Exception as e:
        return f"Error predicting class: {e}"

    try:
        # Find the class name
        for class_info in classes:
            if class_info.get('count') == class_num:
                class_name = class_info.get('name')
                break
    except KeyError as e:
        return f"Key error: {e}"
    except Exception as e:
        return f"Error finding class name: {e}"
    
    return class_name or "Could not identify shape."

if __name__ == "__main__":
    DrawingClassifier()

    model = load_model_details('t')
    if model is not None:
        data = model['data']
        classes = model['classes']
        clf = model['clf']
        proj_name = model['proj_name']
        image = r'predictshape.png'
        prediction = predict(image, clf, classes)
        print(prediction)
    else:
        print("Something went wrong when loading the model. See traceback for more details.")