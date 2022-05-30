import os
import shutil
import git
import urllib.request
from workspace_paths import paths, TFOD_DIR_NAME, PRETRAINED_MODEL_NAME, PRETRAINED_MODEL_URL


def downloads_and_installs():
    if not os.path.isdir(TFOD_DIR_NAME):
        os.mkdir(TFOD_DIR_NAME)
    if not os.path.isdir(paths['PROTOC']):
        os.mkdir(paths['PROTOC'])



    # Download and extract the LogoDet-3K dataset, unless it's already downloaded
    if not os.path.isdir(paths['LOGODET_3K']):
        LOGODET_3K_URL = 'http://123.57.42.89/Dataset_ict/LogoDet-3K.zip'
        try:
            print('Downloading LogoDet-3K...')
            urllib.request.urlretrieve(LOGODET_3K_URL, paths['LOGODET_3K'] + '.zip')
            print('Download complete.')
        except urllib.error.URLError:
            print('Unreachable: ' + LOGODET_3K_URL)
        shutil.unpack_archive(paths['LOGODET_3K'] + '.zip', paths['DATASET'])



    # Download and unzip a PIP package that splits datasets into training and testing sets.
    # I install it through GitHub because the package is not updated on PyPi:
    # https://pypi.org/project/annotated-images/
    try:
        import annotated_images
    except ModuleNotFoundError:
        try:
            print('Downloading annotated-images package...')
            git.Repo.clone_from(
                'https://github.com/paul2048/annotated-images',
                paths['ANNOTATED_IMAGES_PACKAGE'])
            print('Download complete.')
        except git.exc.GitCommandError:
            # If the destination path is not empty
            print('annotated-images is already downloaded.')
        # Install the PIP package
        os.system(f'pip install {paths["ANNOTATED_IMAGES_PACKAGE"]}')



    # Download the TensoFlow Object Detection API, unless it's already downloaded
    try:
        import object_detection
    except ModuleNotFoundError:
        # Download the TensoFlow Object Detection API
        try:
            print('Downloading the TensorFlow Object Detection API...')
            git.Repo.clone_from('https://github.com/tensorflow/models', TFOD_DIR_NAME)
            print('Download complete.')
        except git.exc.GitCommandError:
            # If the destination path is not empty
            print('The API is already downloaded.')

        # Download and unzip protobuf
        PROTOC_URL = 'https://github.com/protocolbuffers/protobuf/releases/download/v3.20.0/protoc-3.20.0-win64.zip'
        protoc_zip = os.path.join(paths['PROTOC'], 'protoc.zip')
        try:
            print(f'Downloading protobuf...')
            urllib.request.urlretrieve(PROTOC_URL, protoc_zip)
            print('Download complete.')
        except urllib.error.URLError:
            print('Unreachable: ' + PROTOC_URL)
        shutil.unpack_archive(protoc_zip, paths['PROTOC'])

        os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC'], 'bin'))   
        os.system('cd tf_object_detection/research && '
                'protoc object_detection/protos/*.proto --python_out=. && '
                'copy object_detection\\packages\\tf2\\setup.py setup.py && '
                'python setup.py build && '
                'python setup.py install')
        os.system('cd tf_object_detection/research/slim && '
                'pip install -e .')



    # Download the pretrained model
    if not os.path.exists(os.path.join(paths['PRETRAINED_MODEL_CKPT'])):
        model_tar = os.path.join(paths['MODELS'], 'model.tar.gz')
        try:
            print(f'Downloading pretrained {PRETRAINED_MODEL_NAME}...')
            urllib.request.urlretrieve(PRETRAINED_MODEL_URL, model_tar)
            print('Download complete.')
        except urllib.error.URLError:
            print('Unreachable: ' + PRETRAINED_MODEL_URL)
        shutil.unpack_archive(model_tar, paths['MODELS'])

        # Rename the extracted folder of the pretrained model
        old_dir_name = os.path.join(paths['MODELS'], PRETRAINED_MODEL_NAME)
        new_dir_name = paths['PRETRAINED_MODEL']
        try:
            os.rmdir(paths['PRETRAINED_MODEL'])
        except FileNotFoundError:
            pass
        shutil.move(old_dir_name, new_dir_name)
        # Remove the downloaded archive
        os.remove(model_tar)
        # Rename the "checkpoint" directory (https://stackoverflow.com/a/64159833/7367049)
        os.rename(
            os.path.join(paths['PRETRAINED_MODEL'], 'checkpoint'),
            os.path.join(paths['PRETRAINED_MODEL'], 'checkpoint0'))
