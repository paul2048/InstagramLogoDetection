import os
# Enable GPU for training
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import shutil
import instagram_scraper
import base64
import cv2
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
from annotations.generate_tfrecord import generate_tfrecord
from workspace_paths import paths
from downloads_and_installs import downloads_and_installs

# Downloads necessary resources:
#   1) LogoDet-3k dataset
#   2) "annotated-images" package (used for splitting into train and test sets)
#   3) TensoFlow Object Detection API
#   4) pretrained model
downloads_and_installs()

import annotated_images
from tf_object_detection.research.object_detection.model_main_tf2 import model_main_tf2
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# Repalce the TF object detection training script with the essentially same script,
# but it can be imported rather than being used inside a terminal
if os.path.isfile('model_main_tf2.py'):
    shutil.copyfile(
        'model_main_tf2.py',
        os.path.join(paths['OBJECT_DETECTION'], 'model_main_tf2.py'))

# Update the config file of the pretrained model
config = config_util.get_configs_from_pipeline_file(paths['PIPELINE_CONFIG'])
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(paths['PIPELINE_CONFIG'], 'r') as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

pipeline_config.train_config.batch_size = 16
pipeline_config.train_config.fine_tune_checkpoint_type = 'detection'
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_CKPT'], 'ckpt-0')
pipeline_config.train_input_reader.label_map_path = paths['LABEL_MAP']
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATIONS'], 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = paths['LABEL_MAP']
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATIONS'], 'test.record')]

config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(paths['PIPELINE_CONFIG'], 'wb') as f:
    f.write(config_text)

# The unique logo names from the dataset
LOGOS = sorted([
    brand
    for category in os.listdir(paths['LOGODET_3K'])
    for brand in os.listdir(os.path.join(paths['LOGODET_3K'], category))
])



# Create Flask application
app = Flask(__name__)
CORS(app)
# Create socket and fix CORS errors
socketio = SocketIO(app, cors_allowed_origins='*')

@app.route('/get_logos', methods=['GET'])
def serve_logos():
    return jsonify({'logos': LOGOS})

@app.route('/detect_logos', methods=['POST'])
def detect_logos():
    usernames = request.json['usernames']
    selected_logos = request.json['selectedLogos']

    # Remove the previous splitted data if it exists, and create an empty folder
    if os.path.isdir(paths['LOGODET_3K_FLAT_SUBSET']):
        shutil.rmtree(paths['LOGODET_3K_FLAT_SUBSET'])
    os.mkdir(paths['LOGODET_3K_FLAT_SUBSET'])

    # Create the label map
    print('Creating the label map...')
    label_map = '\n'.join([
        f'item {{\n    id: {i+1}\n    name: "{brand_name}"\n}}'
        for i, brand_name in enumerate(selected_logos)
    ])
    # Write the label map into a file
    with open(paths['LABEL_MAP'], 'w+') as f:
        f.write(label_map)

    # Update the config file to have the correct number of classes
    pipeline_config.model.center_net.num_classes = len(selected_logos)
    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(paths['PIPELINE_CONFIG'], 'wb') as f:
        f.write(config_text)

    # Build the detection model
    detection_model = model_builder.build(model_config=config['model'], is_training=False)

    TRAINSET_PATH = os.path.join(paths['DATASET'], 'train')
    TESTSET_PATH = os.path.join(paths['DATASET'], 'test')
    if os.path.isdir(TRAINSET_PATH):
        shutil.rmtree(TRAINSET_PATH)
    if os.path.isdir(TESTSET_PATH):
        shutil.rmtree(TESTSET_PATH)
    # Flatten the data (using only the logos selected by the user).
    # Loop over the LogoDet-3K brands catagories (Food, Technology, etc.)
    print('Copying the images of the selected logos...')
    for category in os.listdir(paths['LOGODET_3K']):
        brands_path = os.path.join(paths['LOGODET_3K'], category)
        # Loop over the brands (7-Up, Apple, etc.)
        for brand in os.listdir(brands_path):
            if brand in selected_logos:
                brand_path = os.path.join(brands_path, brand)
                # Loop over each file (1.jpg, 1.xml, etc.)
                for file_ in os.listdir(brand_path):
                    source_path = os.path.join(brand_path, file_)
                    # Append the name of the brand at the beginning of the file name
                    # because "1.jpg" (for example) repeats at each brand
                    if file_.endswith('.xml'):
                        with open(source_path) as f:
                            tree = ET.parse(f)
                            root = tree.getroot()
                            xml_filename = root.find('filename')
                            num, extenssion = xml_filename.text.split('.')
                            num = num.split('_')[-1]
                            xml_filename.text = f'{brand}_{num}.{extenssion}'
                        tree.write(source_path)
                    destination_path = os.path.join(paths['LOGODET_3K_FLAT_SUBSET'], f'{brand}_{file_}')
                    shutil.copy(source_path, destination_path)

    # Split the flat dataset into training and testing sets
    print('Splitting the data into train and test sets...')
    annotated_images.split(
        paths['LOGODET_3K_FLAT_SUBSET'],
        output_dir=paths['DATASET'],
        seed=42, ratio=(0.7, 0.3))

    # Generate tfrecord files using the train and test sets
    generate_tfrecord(
        os.path.join(paths['DATASET'], 'train'),
        paths['LABEL_MAP'],
        os.path.join(paths['ANNOTATIONS'], 'train.record'))
    generate_tfrecord(
        os.path.join(paths['DATASET'], 'test'),
        paths['LABEL_MAP'],
        os.path.join(paths['ANNOTATIONS'], 'test.record'))

    # Train the model
    print('Training the model...')
    model_main_tf2(
        pipeline_config_path=paths['PIPELINE_CONFIG'],
        model_dir=paths['PRETRAINED_MODEL'],
        num_train_steps=2000)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(paths['PRETRAINED_MODEL'], 'ckpt-4')).expect_partial()

    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections

    def detect_and_save(img_path):
        """
        Takes a path of an image as input. Detects the logos in that image.
        Returns a numpy array representing the image with the bounding boxes drawn.
        Returns `None` is no logo was detected.
        """

        category_index = label_map_util.create_category_index_from_labelmap(paths['LABEL_MAP'])
        img = cv2.imread(img_path)
        image_np = np.array(img)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        min_score_thresh = 0.4

        if tf.reduce_max(detections['detection_scores']) > min_score_thresh:
            num_detections = int(detections.pop('num_detections'))
            detections = {
                key: value[0, :num_detections].numpy()
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
            return image_np_with_detections
        return None

    args = {
        'media_types': ['image'],
        'maximum': 9999,
        'login_user': None, #####
        'login_pass': None  #####
    }

    insta_scraper = instagram_scraper.InstagramScraper(**args)
    insta_scraper.authenticate_as_guest()
    # insta_scraper.authenticate_with_login()

    for username in usernames:
        user_dir = os.path.join(paths['PHOTOS'], username)
        try:
            os.mkdir(user_dir)
        except FileExistsError:
            pass

        shared_data = insta_scraper.get_shared_data_userinfo(username=username)
        # Generator that includes the data of each instagram post of the current user
        user_images_data_len = shared_data['edge_owner_to_timeline_media']['count']

        for i, item in enumerate(insta_scraper.query_media_gen(shared_data)):
        # for i in range(40):
            print(i)
            photo_path = os.path.join(user_dir, f'{i}.jpg')
            try:
                urllib.request.urlretrieve(item['display_url'], photo_path)
            except urllib.error.URLError:
                print('Unreachable: ' + item['display_url'])

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
                    'logos': selected_logos,
                    # 'originalSrc': None,
                    'originalSrc': 'https://www.instagram.com/p/' + item['shortcode'],
                    # 'likes': 32,
                    'likes': item['edge_media_preview_like']['count'],
                    # Convert seconds to milliseconds because JavaScript uses milliseconds
                    # 'timestamp': 10000000,
                    'timestamp': item['taken_at_timestamp'] * 1000,
                })

            # Emit the progress percentage
            socketio.emit(f'send_progress_{username}', {
                'progress': ((i+1) / user_images_data_len) * 100,
                # 'progress': f'{(((i+1) / 40) * 100):.2f}',
            })
        print(f'{username} done.')
    return 'good'

if __name__ == '__main__':
    socketio.run(app)
