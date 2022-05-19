import sys
import os
import model_utils
import preprocessor
from datetime import datetime

def parse_args():
    args = sys.argv[1:]

    ret = dict()

    valid_keys = {'input', 'output', 'model'}

    for a in args:
        key, val = a.split('=')
        if key in ret:
            raise Exception('The same information cannot be specified twice.')
        if key in valid_keys:
            ret[key] = val

    return ret

def get_all_images(path):
    """
    Returns list of all images in given directory
    """
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]

    files = filter(lambda f : f.split('.')[-1] in ['jpg', 'png', 'tiff', 'tif'], files)

    return list(files)

def LOG(message):
    init = '\u001b[33mLOG: \u001b[0m'
    print(init+message)

if __name__ == '__main__':
    args = parse_args()
    
    # Load paths for input images
    if 'input' not in args:
        raise Exception('The path for input images has to be specified')
    
    input_imgs = get_all_images(args['input'])
    LOG(f'{len(input_imgs)} input images have been found')

    # Load paths for output images
    if 'output' not in args:
        raise Exception('The path for output images has to be specified')
    if not os.path.exists(args['output']):
        os.makedirs(args['output'])
    
    # Check the save-model path is specified
    if 'model' not in args:
        raise Exception('The path for model has to be specified')
    
    # Load the model
    LOG('Loading model')
    model = model_utils.load_model(args['model'])
    
    # Compile model
    LOG('Compiling model')
    model_utils.compile_model(model, 1.)

    # Run inference
    for inp in input_imgs:
        LOG(f'Running inference on image {inp}')
        img = preprocessor.load_image(os.path.join(args['input'], inp))
        predicted = model_utils.run_inference(model, img)
        preprocessor.save_image(predicted, os.path.join(args['output'], inp))
    
    LOG('Finished successfuly')