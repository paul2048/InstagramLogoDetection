import os

# Type your instagram login data here
INSTAGRAM_USERNAME = None
INSTAGRAM_PASSWORD = None

TFOD_DIR_NAME = 'tf_object_detection'
PRETRAINED_MODEL_NAME = 'centernet_resnet50_v1_fpn_512x512_coco17_tpu-8'
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
