import os

# Type your instagram login data here
INSTAGRAM_USERNAME = None
INSTAGRAM_PASSWORD = None

# The score (between 0 and 1) of the detection
MIN_DETECTION_SCORE = 0.6
# Number of examples in a minibatch
BATCH_SIZE = 1
# The number of times to pass through each minibatch
TRAIN_STEPS = 5000
# The number of maximum images to check for each Instagram account
MAX_IMGS_PER_USER = 100

TFOD_DIR_NAME = 'tf_object_detection'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v2_512x512_coco17_tpu-8.tar.gz'
PRETRAINED_MODEL_NAME = PRETRAINED_MODEL_URL.split('/')[-1].rstrip('.tar.gz') 

paths = {
    'DATASET': 'dataset',
    'EXTRA_DATA': 'extra_data',
    'ANNOTATED_IMAGES_PACKAGE': os.path.join('dataset', 'annotated-images'),
    'ANNOTATIONS': 'annotations',
    'LOGODET_3K': os.path.join('dataset', 'LogoDet-3K'),
    'LOGODET_3K_FLAT_SUBSET': os.path.join('dataset', 'LogoDet_3K_flat_subset'),
    'PHOTOS': 'photos',
    'PROTOC': 'protoc',
    'MODELS': 'models',
    'PRETRAINED_MODEL': os.path.join('models', 'pretrained'),
    'PRETRAINED_MODEL_CKPT': os.path.join('models', 'pretrained', 'checkpoint0'),
    'TRAINED_MODELS': os.path.join('models', 'trained'),
    'OBJECT_DETECTION': os.path.join(TFOD_DIR_NAME, 'research', 'object_detection'),
}
