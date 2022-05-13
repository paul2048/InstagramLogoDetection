# Instagram Logo Detection
![demo](demo.gif)

## About
Instagram Logo Detection is a Flask + React web application that downloads photos from the selected Instagram usernames and tries to detect the specified logos in each photo using.  

The photos are downloaded with [Instagram Scraper](https://github.com/arc298/instagram-scraper)'s scraped data, inputted into a Tensorflow object detection model, and the photos with any detection will be sent to the frontend. Each photo detection is sent once the detection happens, one at a time, because the WebSocket protocol is used instead of HTTP.

The range of logos you can choose from is based on the [LogoDet-3K](https://github.com/Wangjing1551/LogoDet-3K-Dataset) dataset which has 3000 unique logos. For each pair of selected logos you choose, a new model will be trained. If the same set of logos is used again (even if you use different Instagram usernames), the model will be restored from a checkpoint and the detection process starts straight away.  

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Issues](https://img.shields.io/github/issues/paul2048/InstagramLogoDetection?style=flat)](https://github.com/paul2048/InstagramLogoDetection/issues)

## Installation
Make sure you have Python 3 and Node installed on your machine, then:
1. Clone this repo: `git clone https://github.com/paul2048/InstagramLogoDetection`
2. Create the virtual environment: `python3 -m venv env`
3. Install the server's required packages: `pip3 install requirements.txt`
4. Install the frontend packages inside the "frontend" folder: `npm i`
5. (OPTIONAL) If you have a dedicated GPU, [install CUDA and CUDNN](https://medium.com/geekculture/install-cuda-and-cudnn-on-windows-linux-52d1501a8805) if you want faster training

## Settings
### `workspace_path.py`
Set `INSTAGRAM_USERNAME` and `INSTAGRAM_PASSWORD` to YOUR Instagram username and password. Your account is needed because Instagram Scraper will have issues scraping without an account being logged in.

If you want to get more detections with lower confidence, set `MIN_DETECTION_SCORE` lower. If you want fewer detections, but with higher confidence, set `MIN_DETECTION_SCORE` higher. For example, if we set it to `0.6`, only the bounding boxes with the confidence of >= 60% will be drawn to the photo and sent to the client-side.  

`BATCH_SIZE` is set to `1` by default, which means that Stochastic Gradient Descent (SGD) will run during training. I found that larger batch sizes (i.e. 4, 8, or 16) will help trian a better model, but on my local GPU (GeForce RTX 3050 4GB), larger batch sizes do not work because my machine runs out of memory. Use my [Jupyter Notebook](https://github.com/paul2048/LogoDetectionNotebook) on Google Colab to train if you want to use higher batch sizes. 

`TRAIN_STEPS` represents the number of training steps to take for each batch. Set it higher if you feel like your model has a high biased for your selected logos, and lower if your model has a high variance for your selected logos.

![high_bias_high_variance](https://1.cms.s81c.com/sites/default/files/2021-03-03/model-over-fitting.png)

If an Instagram user has a ton of images, you might not want to go through all of them. `MAX_IMGS_PER_USER` tells Instagram Scraper to only scrape the last `1000` images (by default). Setting the value too high and having many Instagram usernames, might restrict scraping on your account for some time (1-2 days) or even **suspend your account**.  

![suspended_account](https://i.imgur.com/EShmiXK.png)

### `extra_data/`
The "extra_data" directory can be used to add extra labelled images if your model performs poorly using only LogoDet-3K's data.  

```
extra_data
├── Apple
│   └── image1.jpg
│   └── image1.xml
│   └── image2.jpg
│   └── image2.xml
|   └── ...
└── ASUS
│   └── image1.jpg
│   └── image1.xml
│   └── image2.jpg
│   └── image2.xml
|   └── ...
```

## Run the app
Inside the project's folder run `env\Scripts\Activate.ps1; set FLASK_APP=app.py; flask run` (Windows Powershell).  

Go inside the "frontend" folder and run `npm start`.
