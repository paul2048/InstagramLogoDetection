import os

TFOD_DIR_NAME = 'tf_object_detection'
PRETRAINED_MODEL_NAME = 'centernet_resnet101_v1_fpn_512x512_coco17_tpu-8'
paths = {
    'DATASET': 'dataset',
    'ANNOTATED_IMAGES_PACKAGE': os.path.join('dataset', 'annotated-images'),
    'ANNOTATIONS': 'annotations',
    'LOGODET_3K': os.path.join('dataset', 'LogoDet-3K'),
    'LOGODET_3K_FLAT_SUBSET': os.path.join('dataset', 'LogoDet_3K_flat_subset'),
    'PHOTOS': 'photos',
    'PROTOC': 'protoc',
    'PRETRAINED_MODEL': os.path.join('models', 'pretrained'),
    'PRETRAINED_MODEL_CKPT': os.path.join('models', 'pretrained', 'checkpoint0'),
    'TRAINED_MODEL': os.path.join('models', 'trained'),
    'RESEARCH': os.path.join(TFOD_DIR_NAME, 'research'),
    'OBJECT_DETECTION': os.path.join(TFOD_DIR_NAME, 'research', 'object_detection'),
    'MODELS': 'models',
    'LABEL_MAP': os.path.join('annotations', 'label_map.pbtxt'),
    'PIPELINE_CONFIG': os.path.join('models', 'pretrained', 'pipeline.config'),
}
