from classifier import CaripelaClassifier
import json, base64, os, io, datetime
from PIL import Image
from StringIO import StringIO


clasificados = CaripelaClassifier()

MOBILENET_TRAINING = './test_data/22-08-2018-00_50/trainingData.json'
MOBILENET_TEST = './test_data/22-08-2018-00_50/testData.json'

MOBILENET_ALIGNED_TRAINING = './test_data/images/training_aligned_224.json'
MOBILENET_ALIGNED_TEST = './test_data/images/test_aligned_224.json'

FACENET_TRAINING = './test_data/images/training_160.json'
FACENET_TEST = './test_data/images/test_160.json'
TRAINING_224 = '224x224/training'
TEST_224 = '224x224/test'
TRAINING_160 = '160x160/training'
TEST_160 = '160x160/test'


def load_json_samples(filename):
    with open(filename) as f:
        data = json.load(f)

    return data


""" Iterate a test folder (facenet) and create a json data test from the images in the folder"""


def load_samples_folder():
    result = {}
    for path, subdirs, files in os.walk('./test_data/images/224x224/training_aligned'):
        for dir in subdirs:
            folder = os.path.join(path, dir)
            for filename in os.listdir(folder):
                # with open(os.path.join(folder, filename)) as f:
                if dir not in result:
                    result.__setitem__(dir, [])
                # content = f.read()
                img = Image.open(os.path.join(folder, filename))

                imgByteArr = io.BytesIO()
                img.save(imgByteArr, format='JPEG', quality=100)
                imgByteArr = imgByteArr.getvalue()

                img_str = base64.b64encode(imgByteArr)
                result[dir].append(img_str)

    with open('./test_data/images/training_aligned_224.json', 'w') as outfile:
        json.dump(result, outfile)


""" Parse an image in base64 string and return a PIL image"""


def process_base64(base64str):
    image_data = base64.b64decode(base64str)
    ima = Image.open(StringIO(image_data))
    # ima = ImageOps.fit(ima, (160, 160))
    return ima


""" Export trainig data from json to folder images"""


def create_folder(samples_purpouse, class_id):
    file_path = './test_data/images/' + samples_purpouse + '/' + class_id + '/'
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    return file_path


def save_images(data, folder_name):

    for class_id, samples in data.iteritems():
        for index, img_str in enumerate(samples):
            image = process_base64(img_str)
            file_path = create_folder(folder_name, class_id)
            image.save(file_path + str(index) + '.jpg')


def train(data):
    print ('start adding samples', datetime.datetime.now())

    for class_id, samples in data.iteritems():
        for img_str in samples:
            print('process_samples > sample added for label ', class_id)
            clasificados.add_sample(img_str, class_id)

    print ('end adding samples', datetime.datetime.now())

    print ('start training samples', datetime.datetime.now())
    clasificados.train()
    print ('end training samples', datetime.datetime.now())

def predict_samples(data):
    for class_id, samples in data.iteritems():
        for index, img_str in enumerate(samples):
            print('testing image from class', class_id)
            response = clasificados.predict(img_str)
            if float(response['confidence']) < 0.9:
                print('MENOR ' + str(index))


def test_mobilenet():
    training_samples = load_json_samples(MOBILENET_ALIGNED_TRAINING)
    train(training_samples)

    test_samples = load_json_samples(MOBILENET_ALIGNED_TEST)
    print ('start predicting ', datetime.datetime.now())
    predict_samples(test_samples)
    print ('end predicting ', datetime.datetime.now())

def test_facenet():
    training_samples = load_json_samples(FACENET_TRAINING)
    train(training_samples)

    test_samples = load_json_samples(FACENET_TEST)
    predict_samples(test_samples)


def process_json_to_images():
    training_samples = load_json_samples(MOBILENET_TRAINING)
    save_images(training_samples, TRAINING_224)

    test_samples = load_json_samples(MOBILENET_TEST)
    save_images(test_samples, TEST_224)



test_mobilenet()