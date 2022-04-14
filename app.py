from flask import Flask, jsonify, request, send_file
from flask_cors import CORS # Comment this on deployment
from flask_socketio import SocketIO
from pprint import pprint
import instagram_scraper
import urllib.request
import os
import base64
import cv2


app = Flask(__name__)
CORS(app) # Comment this on deployment
# Create socket and fix CORS errors
socketio = SocketIO(app, cors_allowed_origins='*')

LOGOS = ['Corsair', 'Apple', 'AMD', 'Nvidia']
PHOTOS_DIR = os.path.join('photos')

@app.route('/get_logos', methods=['GET'])
def serve_logos():
    return jsonify({'logos': LOGOS})

@app.route('/detect_logos', methods=['POST'])
def detect_logos():
    usernames = request.json['usernames']
    logos = request.json['logos']
    usernames = ['test_123456722']
    args = {
        'media_types': 'image',
        'maximum': 9999,
    }

    insta_scraper = instagram_scraper.InstagramScraper(**args)
    # insta_scraper.authenticate_as_guest()
    # insta_scraper.authenticate_with_login()

    for username in usernames:
        user_dir = os.path.join(PHOTOS_DIR, username)
        try:
            os.mkdir(user_dir)
        except FileExistsError:
            pass

        # shared_data = insta_scraper.get_shared_data_userinfo(username=username)
        # for i, item in enumerate(insta_scraper.query_media_gen(shared_data)):
        for i in range(6):
            photo_path = os.path.join(user_dir, f'{i}.jpg')
            # try:
            #     urllib.request.urlretrieve(item['display_url'], photo_path)
            # except urllib.error.URLError:
            #     print('Unreachable: ' + item['display_url'])
            # Load the image with OpenCV and convert it into text
            photo = cv2.imread(photo_path)
            _, photo = cv2.imencode('.jpg', photo)
            text_photo = base64.b64encode(photo).decode('utf-8')
            # Emit the image to the front end
            socketio.emit('receive_detection_image', {
                'username': username,
                'src': 'data:image/jpg;base64,' + text_photo,
                'logos': ['AMD', 'Apple'],
                'originalSrc': None, # 'https://www.instagram.com/p/' + shared_data['shortcode']
                'likes': 32, # shared_data['edge_media_to_comment']['count']
                # Convert seconds to milliseconds because JavaScript uses milliseconds
                'timestamp': 10000000 # shared_data['taken_at_timestamp'] * 1000
            })
    return 'good'

if __name__ == '__main__':
    socketio.run(app)
