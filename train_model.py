import sys
import os
import model_utils
import preprocessor
from datetime import datetime
import tensorflow as tf

def parse_args():
    args = sys.argv[1:]

    ret = dict()

    valid_keys = {'training-images', 'load-model', 'save-model', 'steps-per-epoch', 'epochs', 'runs', 'learning-rate', 'batch-size', 'prefetch'}

    for a in args:
        key, val = a.split('=')
        if key in ret:
            raise Exception('The same information cannot be specified twice.')
        if key in valid_keys:
            ret[key] = val
    
    if 'steps-per-epoch' not in ret:
        raise Exception('`steps-per-epoch` must be specified')

    if 'epochs' not in ret:
        raise Exception('`epochs` must be specified')

    for key in ['steps-per-epoch', 'epochs', 'runs', 'batch-size', 'prefetch']:
        if key in ret:
            ret[key] = int(ret[key])
    
    for key in ['learning-rate']:
        if key in ret:
            ret[key] = float(ret[key])


    return ret

def get_all_annotated_images(path):
    """
    Returns list of all images in given directory which have also annotated_* file associated
    """
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]

    files = filter(lambda f : f.split('.')[-1] in ['jpg', 'png', 'tiff', 'tif'], files)
    files = list(files)

    valid = set()

    for f in files:
        if f.startswith('annotated_'):
            continue
        
        if 'annotated_'+f in files:
            valid.add(f)
    
    return list(valid)

def LOG(message):
    init = '\u001b[33mLOG: \u001b[0m'
    print(init+message)

def main(args):
    LOG(f'Using tensorflow version {tf.__version__}')

    if 'batch-size' not in args:
        args['batch-size'] = 2
    if 'prefetch' not in args:
        args['prefetch'] = 2
    if 'learning-rate' not in args:
        args['learning-rate'] = 0.001

    # Load paths for training images
    if 'training-images' not in args:
        raise Exception('The path for training images has to be specified')

    train_paths = get_all_annotated_images(args['training-images'])

    LOG(f'{len(train_paths)} training images have been found')
    print(train_paths)

    # Check the save-model path is specified
    if 'save-model' not in args:
        raise Exception('The path for saving model has to be specified')
    
    LOG_FILE = os.path.join(args['save-model'],'log.txt')

    # Create log file in the save-model folder
    if not os.path.exists(args['save-model']):
        os.makedirs(args['save-model'])
    with open(LOG_FILE, 'w') as f:
        f.write(f'Starting run at {datetime.now()}')
        LOG('Created log file')

    # Create model and load weights
    
    if 'load-model' in args:
        LOG('Loading model')
        model = model_utils.load_model(args['load-model'])
    else:
        LOG('Creating new model')
        model = model_utils.build_new_model()
        

    # Load the training dataset
    LOG('Loading images')
    dataset, class_counts = model_utils.create_dataset(args['training-images'], train_paths)
    LOG(f'There are {class_counts[0]} entries for yellow and {class_counts[1]} entries for cyan')

    # Compile model
    LOG('Compiling model')
    model_utils.compile_model(model, classes_ratio=class_counts[0]/class_counts[1], learning_rate=args['learning-rate'])

    # Fit model
    LOG('Training model')

    if 'runs' in args:
        for i in range(1,args['runs']+1):
            LOG(f"Run {i}/{args['runs']}")
            model.fit(dataset.repeat().batch(args['batch-size']).prefetch(args['prefetch']), steps_per_epoch=args['steps-per-epoch'], epochs=args['epochs'])
            LOG(f'Saving model {i}')
            model.save(args['save-model']+f'_{i}', include_optimizer=False)
    else:
        model.fit(dataset.repeat().batch(args['batch-size']).prefetch(args['prefetch']), steps_per_epoch=args['steps-per-epoch'], epochs=args['epochs'])

    # Save model
    LOG('Saving final model...')
    model.save(args['save-model'], include_optimizer=False)
    LOG('Model saved')

if __name__ == '__main__':
    args = parse_args()
    main(args)