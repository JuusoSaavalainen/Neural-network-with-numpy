import tkinter as tk
import tkinter.messagebox
import cv2
import utility as utils
import dataformat as dataformat
import numpy as np
import pickle


def save_image():
    x, y = (28, 28)
    image = np.zeros((x, y), dtype=np.uint8)

    for item in canvas.find_all():
        coords = canvas.coords(item)
        x0, y0, x1, y1 = coords
        for i in range(0, x):
            for j in range(0, y):
                if x0 <= j*10 <= x1 and y0 <= (27-i)*10 <= y1:
                    image[i][j] = 255

    image = np.flip(image, axis=0)
    image = cv2.blur(image, (2, 2))
    image = image.reshape(784, 1)
    image = dataformat.normalize_zero_one(image)
    image = np.array(image)

    # plt.imshow(image.reshape(28, 28), cmap='gray')
    # plt.show()
    # un comment to see as in mnist and with smoothening applied.

    make_guess(image)


def make_guess(image):
    reset()
    nn = utils.NeuralNetwork([784, 256, 128, 64, 10])
    params = nn.load_params()
    activations = nn.forwardprop(image, 'relU')
    predictedout = activations[f'A4']

    # want to see the outputlayer as whole? uncomment this
    # print(predictedout)

    act_pred = np.argmax(predictedout)
    display_number(act_pred)


def load_dict():
    with open('src/model/model98.pickle', 'rb') as f:
        params = pickle.load(f)
    return params


number_labels = []


def reset():
    for label in number_labels:
        label.destroy()
    number_labels.clear()
    canvas.delete("all")


def display_number(num):
    number_label = tk.Label(root, text=f"{num}")
    number_label.pack()
    number_labels.append(number_label)


def draw(event):
    x, y = (event.x, event.y)
    canvas.create_oval(x-10, y-10, x+10, y+10, fill="black", outline="black")


def display_message():
    tkinter.messagebox.showinfo("About this UI", "Please draw slowly to get better results, and try to write in the center. Since the implementation of drawing is not perfect moving the mouse too fast while drawing will lead to bad results. Also please try to avoid drawing in the corners and walls of the canvas. Drawing too small digits will also impact negatively since the canvas will be smoothened out when runned on the model")


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Digit regocniti")

    canvas = tk.Canvas(root, width=280, height=280, bg="white")
    canvas.pack()

    canvas.bind("<B1-Motion>", draw)

    save_button = tk.Button(root, text="Make Ai guess", command=save_image)
    save_button.pack()

    message_button = tk.Button(root, text="UI INFO", command=display_message)
    message_button.pack()

    root.mainloop()
