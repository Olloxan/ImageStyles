import numpy as np
import matplotlib.pyplot as plt

image = np.zeros((64, 64,3))
fig, ax = plt.subplots()
im = ax.imshow(image)

def displayImageION(image):   
    if not plt.isinteractive():
        plt.ion()
    im.set_data(image)
    fig.canvas.draw()
    plt.pause(0.1)


def showImage(imgArray):    
    if plt.isinteractive():        
        plt.ioff()
    
    plt.imshow(imgArray)
    plt.show()   