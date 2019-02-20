from tkinter import *

import cv2
from six.moves import cPickle as pickle
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np


class Window(Frame):
    image_size = 32  # width and height of image
    num_of_channels = 3  # number of channels in an images
    prediction_label = -1  # prediction from classifier
    label_names = -1  # names of label indices
    model_saver = -1  # model to be loaded

    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):
        # parameters that you want to send through the Frame class.
        Frame.__init__(self, master)

        # reference to the master widget, which is the tk window
        self.master = master

        # with that, we want to then run init_window, which doesn't yet exist
        self.init_window()

        # load label names
        all_data = pickle.load(open('CIFAR_100_processed.pickle', 'rb'))
        self.label_names = all_data['label_names']

        del all_data

    # Creation of init_window
    def init_window(self):
        # changing the title of our master widget
        self.master.title("Object Classification")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        # creating a menu instance
        menu = Menu(self.master)
        self.master.config(menu=menu)

        # create the file object)
        file = Menu(menu)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        file.add_command(label="Exit", command=self.client_exit)

        # added "file" to our menu
        menu.add_cascade(label="File", menu=file)

        topFrame = Frame(self)
        topFrame.pack()

        button = Button(topFrame, text="Choose Image", fg="black", command=self.showImg)
        button.pack(side=BOTTOM)

        self.prediction_label = Label(self, text="", font=("Helvetica", 16))
        self.prediction_label.pack()
        # importing best model
        self.model_saver = tf.train.import_meta_graph('./best_model/saved_model/model.ckpt.meta')

    # a method to normalize values to range [-1,1]
    def sharpen(self, img):
        # img = self.rgb2gray(img)  # RGB to greyscale
        kernel = (-1 / 256.0) * np.array(np.asarray(
            [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, -476, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]))
        img = np.array(img)
        img = cv2.filter2D(img, -1, kernel)
        return img

    def rgb2gray(self, img):
        return np.dot(np.array(img, dtype='float32'), [0.299, 0.587, 0.114])

    # a method to return dataset in format [num_images,image_width,image_height,num_channels]
    def formatForFeedForward(self, dataset):
        dataset = np.reshape(dataset,
                             (-1, self.image_size, self.image_size, self.num_of_channels)).astype(np.float32)
        return dataset

    # a method to classify the image
    def classifyImages(self, img):
        img = self.formatForFeedForward(img)
        with tf.Session() as sess:
            # restore and feed forward
            self.model_saver.restore(sess, tf.train.latest_checkpoint('./best_model/saved_model/'))
            graph = sess.graph
            one_input = graph.get_tensor_by_name("tf_inputs:0")
            fc_keep_prob = graph.get_tensor_by_name("fully_connected_keep_prob:0")
            conv_keep_prob = graph.get_tensor_by_name("conv_keep_prob:0")

            is_training = graph.get_tensor_by_name("is_training:0")
            prediction = graph.get_tensor_by_name("tf_predictions:0")

            # print([node.name for node in graph.as_graph_def().node])

            feed_dict = {one_input: img, fc_keep_prob: 1, conv_keep_prob: 1, is_training: False}
            pred = sess.run(
                [prediction], feed_dict=feed_dict)
            max_class = np.argmax(pred)
            prediction_str = self.label_names[max_class]
            return prediction_str

    # a method to show image and preprocess it and classify it
    def showImg(self):
        filename = askopenfilename()
        print(filename)
        im = Image.open(filename)
        image_view_size = 200
        resized = im.resize((image_view_size, image_view_size), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(resized)
        im = im.resize((self.image_size, self.image_size), Image.ANTIALIAS)
        im_norm = self.sharpen(im)

        img = Label(self, image=render, width=image_view_size, height=image_view_size)
        img.image = render
        img.place(x=200, y=100)
        prediction = self.classifyImages(im_norm).decode("utf-8")
        self.showText(prediction)

    # a method to show prediction
    def showText(self, message="hello"):
        self.prediction_label.config(text="Prediction: " + message)

    def client_exit(self):
        exit()


# root window created. Here, that would be the only window, but
# you can later have windows within windows.
root = Tk()

root.geometry("600x400")

# creation of an instance
app = Window(root)

# mainloop
root.mainloop()
