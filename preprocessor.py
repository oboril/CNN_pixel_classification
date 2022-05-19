import numpy as np
from matplotlib import pyplot as plt
import skimage
from skimage.exposure import equalize_adapthist, adjust_gamma
from skimage.filters import median
from skimage.morphology import disk
from scipy.optimize import curve_fit
from skimage.color import hsv2rgb, rgb2gray

PX_PER_UM = 1944/900

def get_avg_background(img):
    Nx, Ny = 8, 6
    stepx, stepy = img.shape[1]//Nx, img.shape[0]//Ny
    medians = np.zeros((Nx, Ny), dtype=float)
    coordsX = np.arange(stepx/2, stepx*Nx, stepx, dtype=float)
    coordsY = np.arange(stepy/2, stepy*Ny, stepy, dtype=float)
    for x in range(Nx):
        for y in range(Ny):
            medians[x,y] = np.median(img[y*stepy:y*stepy+stepy, x*stepx:x*stepx+stepx])
    coordsX = np.tile(coordsX, (Ny,1)).T
    coordsY = np.tile(coordsY, (Nx,1))
    xdata = np.stack([coordsX.flatten(), coordsY.flatten()], -1)

    def background_fun(XY, a,b,c,d,e,f):
        x, y = XY.T
        return a+b*x+c*y+d*x**2+e*x*y+f*y**2

    fit, covar = curve_fit(background_fun, xdata, medians.flatten(), p0=[0]*6)

    coordsX = np.tile(np.arange(0,img.shape[1], dtype=float), (img.shape[0],1)).T
    coordsY = np.tile(np.arange(0,img.shape[0], dtype=float), (img.shape[1],1))
    xdata = np.stack([coordsX.flatten(), coordsY.flatten()], -1)
    background = np.reshape(background_fun(xdata, *fit), img.shape[::-1]).T
    return background - np.mean(background)

def equalize(img):
    """
    Returns image with equalized background, contrast and gamma
    """
    global PX_PER_UM

    # equalize background
    img = img-get_avg_background(img)
    
    img = np.clip(img, -1,1)

    # adaptive contrast enhancement
    img = equalize_adapthist(img, clip_limit=0.005, kernel_size=(200*PX_PER_UM))

    # normalize histogram so that avg value is approx 0.5
    gamma = -1/np.log2(np.mean(img))
    img = adjust_gamma(img, gamma)

    return img

def load_image(file_path, preprocess = True):
    """
    Returns specified image as f32 array, if preprocess=True also equalizes the image using `equalize`
    """

    img = skimage.io.imread(file_path)
    img = rgb2gray(img)
    img = np.array(img, dtype=np.float32)/255.
    if preprocess:
        img = equalize(img).astype(np.float32)
    return img

def display_image(img, grayscale=True, size = (16,12)):
    """
    Returns plot of the image
    """
    fig, ax = plt.subplots(figsize=size)
    img = ax.imshow(img, cmap = 'gray' if grayscale else None, aspect='equal')
    plt.axis('off')
    return img

def load_annotated(file_path):
    """
    Loads annotated image as mask of classes, returns array of u8:
    0 = no class
    1 = yellow
    2 = cyan
    """
    img = plt.imread(file_path)
    img = np.array(img, dtype=np.uint8)

    r,g,b = np.transpose(img, (2,0,1))

    mask = np.where((r > 150) & (g > 150) & (b < 100), 1, 0)
    mask += np.where((r < 100) & (g > 150) & (b > 150), 2, 0)

    return mask.astype(np.uint8)

def color_image(img, predicted_prob, value=1):
    """
    Overlays the grayscale `img` with color depending on class probabilities
    0 = yellow
    1 = cyan
    """
    sat = np.abs(predicted_prob-0.5)*2
    hue = 60/360 + 120/360*predicted_prob
    #print(sat.size, hue.size, img.flatten().shape)
    hsv = np.stack([hue, sat, img*value + (1-value)/2], axis=2)
    ret = hsv2rgb(hsv)
    return np.resize(ret, img.shape+(3,))

def save_image(img, path):
    plt.imsave(path, (img*255).astype(np.uint8), cmap='gray')