from scipy.io import loadmat
from datetime import datetime
import os, sys, logging
from PIL import Image
import numpy as np
# import cv2
from pathlib import Path

def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def get_meta(mat_path, db):
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    return full_path, dob, gender, photo_taken, face_score, second_face_score, age


def load_data(mat_path):
    d = loadmat(mat_path)

    return d["image"], d["gender"][0], d["age"][0], d["db"][0], d["img_size"][0, 0], d["min_score"][0, 0]


def mk_dir(dir):
    try:
        os.mkdir( dir )
    except OSError:
        pass

def load_train_data(csv_path, image_root, image_size):
    """
    csv_path: CSVファイルのパス
    画像のルートパス
    
    教師データを取得する
    教師データの情報をリストしたCSVを読み込み画像のパスと正解を受け取る
    """
    
    sys.path.append('module')
    
    # CSV読み込み
    import csv
    csv_file = open("./data/data.csv", "r", encoding="ms932")
    f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
    
    # 1行目はヘッダー情報のため削除
    next(f)
        
    # 画像のパスをリストで取得
    images_path = []
    
    # 性別ラベルを取得する
    genders = []
    
    # 年齢ラベルを取得する
    ages = []
    
    for r in f:
        images_path.append(r[0])
        genders.append(r[1])
        ages.append(r[2])
    
    # 画像データをリストで取得
    images = []
    
    for x in images_path:
        image_path = image_root + x + ".jpg"
        image = Image.open(image_path)
        # image = cv2.imread(image_path)
        
        if image != None:
            image = image.resize((image_size, image_size))
            
            img_array = np.asarray(image)
               
            images.append(img_array)
    
    return np.array(images), np.array(genders), np.array(ages)

def mkoutdir():
    output_path = Path(__file__).resolve().parent.joinpath("/output")
    print(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
def download_weiths():
    sys.path.append('train')
    from keras.utils.data_utils import get_file
    pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"
    modhash = "fbe63257a054c1c5466cfd7bf14646d6"
    weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="pretrained_models",
                                   file_hash=modhash, cache_dir=os.path.dirname(os.path.abspath(__file__)))
    
    return weight_file
