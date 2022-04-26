from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
from pprint import pprint
import instagram_scraper
import urllib.request
import os
import base64
import cv2
import git
import zipfile
import tensorflow as tf


TFOD_DIR_NAME = 'tf_object_detection'
paths = {
    'PHOTOS': 'photos',
    'PROTOC': 'protoc',
    'RESEARCH': os.path.join(TFOD_DIR_NAME, 'research'),
    'OBJECT_DETECTION': os.path.join(TFOD_DIR_NAME, 'research', 'object_detection'),
    'MODEL': 'logo_detection_model',
    'LABEL_MAP': os.path.join('logo_detection_model', 'label_map.pbtxt'),
    'PIPELINE_CONFIG': os.path.join(TFOD_DIR_NAME, 'pipeline.config'),
}

# if not os.path.isdir(TFOD_DIR_NAME):
#     os.mkdir(TFOD_DIR_NAME)
# if not os.path.isdir(paths['PROTOC']):
#     os.mkdir(paths['PROTOC'])

# try:
#     print('Downloading the TensorFlow Object Detection API...')
#     git.Repo.clone_from('https://github.com/tensorflow/models', TFOD_DIR_NAME)
#     print('Download complete.')
# except git.exc.GitCommandError:
#     # If the destination path is not empty
#     print('The API is already downloaded.')

# PROTOC_URL = 'https://github.com/protocolbuffers/protobuf/releases/download/v3.20.0/protoc-3.20.0-win64.zip'
# try:
#     protoc_zip_path = os.path.join(paths['PROTOC'], 'protoc.zip')
#     urllib.request.urlretrieve(PROTOC_URL, protoc_zip_path)
# except urllib.error.URLError:
#     print('Unreachable: ' + PROTOC_URL)

# with zipfile.ZipFile(protoc_zip_path, 'r') as f:
#     f.extractall(paths['PROTOC'])

# os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC'], 'bin'))
# os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC'], 'bin'))   
# os.system('cd tf_object_detection/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection/packages/tf2/setup.py setup.py && python setup.py build && python setup.py install')
# os.system('cd tf_object_detection/research/slim && pip install -e .')

from object_detection.builders import model_builder
from object_detection.utils import config_util

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(os.path.join(paths['MODEL'], 'pipeline.config'))
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['MODEL'], 'ckpt-4')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

import cv2 
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

def detect_and_save(img_path):
    """Detects the logos in the image located at the "img_path" path and ......."""

    category_index = label_map_util.create_category_index_from_labelmap(paths['LABEL_MAP'])
    img = cv2.imread(img_path)
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    min_score_thresh = 0.4

    if tf.reduce_max(detections['detection_scores']) > min_score_thresh:
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=min_score_thresh,
            agnostic_mode=False)

        # print('starting saving...')
        return image_np_with_detections
        # plt.savefig(f'detection_{img_path}')
        # print('saving finished!')
    return None


app = Flask(__name__)
CORS(app)
# Create socket and fix CORS errors
socketio = SocketIO(app, cors_allowed_origins='*')

LOGOS = ['Asus', 'Corsair', 'Apple', 'AMD', 'Nvidia']


@app.route('/get_logos', methods=['GET'])
def serve_logos():
    return jsonify({'logos': LOGOS})

@app.route('/detect_logos', methods=['POST'])
def detect_logos():
    usernames = request.json['usernames']
    logos = request.json['selectedLogos']
    args = {
        'media_types': ['image'],
        'maximum': 9999,
    }

    insta_scraper = instagram_scraper.InstagramScraper(**args)
    # insta_scraper.authenticate_as_guest()
    # insta_scraper.authenticate_with_login()

    for username in usernames:
        user_dir = os.path.join(paths['PHOTOS'], username)
        try:
            os.mkdir(user_dir)
        except FileExistsError:
            pass

        # shared_data = insta_scraper.get_shared_data_userinfo(username=username)
        # # Generator that includes the data of each instagram post of the current user
        # user_images_data_len = shared_data['edge_owner_to_timeline_media']['count']
        # print(user_images_data_len)

        # for i, item in enumerate(insta_scraper.query_media_gen(shared_data)):
        for i in range(40):
            print(i)
            photo_path = os.path.join(user_dir, f'{i}.jpg')
            # try:
            #     urllib.request.urlretrieve(item['display_url'], photo_path)
            # except urllib.error.URLError:
            #     print('Unreachable: ' + item['display_url'])

            det_img = detect_and_save(photo_path)
            if det_img is not None:
                cv2.imwrite(f'{photo_path}', det_img)
                # Load the image with OpenCV and convert it into text
                photo = cv2.imread(f'{photo_path}')
                _, photo = cv2.imencode('.jpg', photo)
                text_photo = base64.b64encode(photo).decode('utf-8')
                # Emit the image to the front end
                socketio.emit('receive_detection_image', {
                    'username': username,
                    'src': 'data:image/jpg;base64,' + text_photo,
                    'logos': logos,
                    'originalSrc': None,
                    # 'originalSrc': 'https://www.instagram.com/p/' + item['shortcode'],
                    'likes': 32,
                    # 'likes': item['edge_media_preview_like']['count'],
                    # Convert seconds to milliseconds because JavaScript uses milliseconds
                    'timestamp': 10000000,
                    # 'timestamp': item['taken_at_timestamp'] * 1000,
                })
            else:
                print(f'No logo in {photo_path}')

            # Emit the progress percentage
            socketio.emit(f'send_progress_{username}', {
                # 'progress': ((i+1) / user_images_data_len) * 100,
                'progress': f'{(((i+1) / 40) * 100):.2f}',
            })
        print(f'{username} done.')
    return 'good'

if __name__ == '__main__':
    socketio.run(app)
