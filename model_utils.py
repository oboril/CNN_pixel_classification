import tensorflow as tf
import preprocessor
import numpy as np
import os

IMG_SHAPE = (1944, 2592)
SHAPE = IMG_SHAPE+(1,)

MODEL_LAYERS = 5

def get_loss(classes_ratio):
    """
    Custom binary crossentropy loss for classes:
    0 = no class
    1 = class 1
    2 = class 2

    classes_ratio = # class 1 / # class 2
    """
    ratio = np.sqrt(classes_ratio)

    def custom_loss(y_true, y_pred):
        cl1 = tf.where(tf.equal(y_true, 1), y_pred, 0)
        cl2 = tf.where(tf.equal(y_true, 2), 1-y_pred, 0)
        return - tf.math.log((1-cl1)*0.9+0.1)/ratio - tf.math.log((1-cl2)*0.9+0.1)*ratio

    return custom_loss

def compile_model(model, classes_ratio=1., learning_rate=0.001):
    optimizer = tf.keras.optimizers.Adamax(learning_rate = learning_rate)
    model.compile(optimizer=optimizer, loss=get_loss(classes_ratio))

def build_new_model():
    from tensorflow.keras import models, layers, losses, regularizers

    global SHAPE

    model = models.Sequential()
    model.add(
        layers.Conv2D(
            6, (5,5),
            activation='selu',
            padding='same',
            input_shape=SHAPE,
            kernel_regularizer=regularizers.l2(0.001)
        )
    )
    model.add(layers.MaxPooling2D((5,5),strides=(1,1),padding='same'))
    model.add(
        layers.Conv2D(
            4, (7,7),
            dilation_rate=5,
            activation='selu',
            padding='same',
            kernel_regularizer=regularizers.l2(0.001)
        )
    )
    model.add(layers.MaxPooling2D((5,5),strides=(1,1),padding='same'))
    model.add(layers.Dropout(0.1))
    model.add(
        layers.Conv2D(
            1, (9, 9),
            dilation_rate=15,
            activation='sigmoid', 
            padding='same',
            kernel_regularizer=regularizers.l2(0.001)
        )
    )

    return model

def load_model(path):
    """
    Loads the TF model from the specified path
    """
    return tf.keras.models.load_model(path)

def create_dataset(path, files):
    """
    returns TF dataset and (# class1, # class2)
    """
    global IMG_SHAPE
    NUM_IMAGES = len(files)

    train_img_paths = [os.path.join(path, f) for f in files]
    annotated_imgs_paths = [os.path.join(path, 'annotated_'+f) for f in files]

    train_imgs = list()
    for f in train_img_paths:
        img = preprocessor.load_image(f)
        if img.shape != IMG_SHAPE:
            raise Exception(f'The image {f} has invalid shape {img.shape}')
        train_imgs.append(img)
    
    class1 = 0
    class2 = 0

    annotated_imgs = list()
    for f in annotated_imgs_paths:
        img = preprocessor.load_annotated(f)
        if img.shape != IMG_SHAPE:
            raise Exception(f'The annotated image {f} has invalid shape {img.shape}')
        annotated_imgs.append(img)
        class1 += np.sum(img == 1)
        class2 += np.sum(img == 2)

    
    train_imgs = np.array(train_imgs).reshape((NUM_IMAGES, *IMG_SHAPE, 1))
    annotated_imgs = np.array(annotated_imgs).reshape((NUM_IMAGES, *IMG_SHAPE, 1))

    train_imgs = tf.constant(train_imgs, dtype=tf.float32)
    annotated_imgs = tf.constant(annotated_imgs, dtype=tf.uint8)

    def dataset_generator():
        while True:
            img = np.random.randint(0,NUM_IMAGES)
            rev1 = np.random.randint(0,1)
            rev2 =np.random.randint(0,1)

            if rev1 and rev2:
                yield train_imgs[img, ::-1, ::-1, :], annotated_imgs[img, ::-1, ::-1, :]
            elif rev1 and not rev2:
                yield train_imgs[img, ::-1, :, :], annotated_imgs[img, ::-1, :, :]
            elif not rev1 and rev2:
                yield train_imgs[img, :, ::-1, :], annotated_imgs[img, :, ::-1, :]
            else:
                yield train_imgs[img, :, :, :], annotated_imgs[img, :, :, :]
    
    generator = tf.data.Dataset.from_generator(
        dataset_generator,
        output_types=(tf.float32,tf.uint8),
        output_shapes = (SHAPE, SHAPE)
    )

    return generator, (class1, class2)

def run_inference(model, img):
    global SHAPE

    return model.predict(img.reshape((1,)+SHAPE))[0,:,:,0]