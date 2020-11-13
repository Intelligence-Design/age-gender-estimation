#import pdb; pdb.set_trace()
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import cv2
#from tensorflow.keras.utils import get_file
import tensorflow as tf
#gpus= tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)
#tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")
from src.factory import get_model
from omegaconf import OmegaConf
import os
import dlib

inWidth = 300
inHeight = 300
confThreshold = 0.5

#prototxt = 'face_detector/deploy.prototxt'
#caffemodel = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'

def crop_face(image, xmin, xmax, ymin, ymax, margine=0.4):
    "slice face image"

    h, w = image.shape[0:2]
    hm = int((ymax - ymin) * margine)
    wm = int((xmax - xmin) * margine)

    mxmin = max(xmin - wm, 0)
    mxmax = min(xmax + wm, w)
    mymin = max(ymin - hm, 0)
    mymax = min(ymax + hm, h)

    return image[mymin:mymax, mxmin:mxmax]


def main():
    #import pdb; pdb.set_trace()
    db = 'megaage_asian'
    root_dir = Path(__file__).parent
    data_dir = root_dir.joinpath("data/megaage_asian")
    train_name_file = root_dir.joinpath("data/megaage_asian/list/train_name.txt")
    test_name_file = root_dir.joinpath("data/megaage_asian/list/test_name.txt")
    train_age_file = root_dir.joinpath("data/megaage_asian/list/train_age.txt")
    test_age_file = root_dir.joinpath("data/megaage_asian/list/test_age.txt")

    genders = []
    ages = []
    img_paths = []

    pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
    modhash = '6d7f7b7ced093a8b3ef6399163da6ece'
    weight_file = tf.keras.utils.get_file("EfficientNetB3_224_weights.11-3.44.hdf5", pretrained_model, cache_subdir="pretrained_models", file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))
    #weight_file = 'checkpoints/MobileNetV2_224_weights.23-3.48.hdf5 '
    model_name, img_size = Path(weight_file).stem.split("_")[:2]
    img_size = int(img_size)
    cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
    model = get_model(cfg)
    model.load_weights(weight_file)

    # for face detection
    detector = dlib.get_frontal_face_detector()

    count = 0
    with open(train_name_file) as name_file, open(train_age_file) as age_file:
        for l in name_file:
            #if count > 500:
            #    break
            age = age_file.readline().replace('\n' , '')
            filename = l.replace( '\n' , '' )
            base = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(base, data_dir, "train", filename)
            print(filepath)
            img = cv2.imread(filepath)
            if img is None:
                print(f"skip: {filepath}")
                continue
            input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detected = detector(input_img, 1)

            if len(detected) > 0:
                xmin, ymin, xmax, ymax = detected[0].left(), detected[0].top(), detected[0].right(), detected[0].bottom()
                face_img = crop_face(img, xmin, xmax, ymin, ymax)
                if face_img is not None:
                    cv2.imwrite(f"data/megaage_asian/merged/train_{filename}", face_img)
                    img_paths.append(f"data/megaage_asian/merged/train_{filename}")
                    ages.append(age)

                    # predict gender of the detected face
                    face_img1 = cv2.resize(face_img, (img_size, img_size))
                    face_img2 = np.expand_dims(face_img1, 0)
                    results = model.predict(face_img2)
                    gender = 0 if results[0][0][0] < 0.5 else 1
                    genders.append(gender)
                    print(f"data/megaage_asian/merged/train_{filename}")
            #count += 1

    with open(test_name_file) as name_file, open(test_age_file) as age_file:
        for l in name_file:
            #if count > 500:
            #    break
            age = age_file.readline().replace('\n' , '')
            filename = l.replace( '\n' , '' )
            base = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(base, data_dir, "test", filename)
            print(filepath)
            img = cv2.imread(filepath)
            if img is None:
                print(f"skip: {filepath}")
                continue
            input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detected = detector(input_img, 1)

            if len(detected) > 0:
                xmin, ymin, xmax, ymax = detected[0].left(), detected[0].top(), detected[0].right(), detected[0].bottom()
                face_img = crop_face(img, xmin, xmax, ymin, ymax)
                if face_img is not None:
                    cv2.imwrite(f"data/megaage_asian/merged/test_{filename}", face_img)
                    img_paths.append(f"data/megaage_asian/merged/test_{filename}")
                    ages.append(age)

                    # predict gender of the detected face
                    face_img1 = cv2.resize(face_img, (img_size, img_size))
                    face_img2 = np.expand_dims(face_img1, 0)
                    results = model.predict(face_img2)
                    gender = 0 if results[0][0][0] < 0.5 else 1
                    genders.append(gender)
                    print(f"data/megaage_asian/merged/test_{filename}")

    outputs = dict(genders=genders, ages=ages, img_paths=img_paths)
    output_dir = root_dir.joinpath("meta")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir.joinpath(f"{db}.csv")
    df = pd.DataFrame(data=outputs)
    df.to_csv(str(output_path), index=False)


if __name__ == '__main__':
    main()
