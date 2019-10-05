# import matplotlib.pyplot as plt
import face_recognition
import numpy as np
import sklearn
import pickle
from face_recognition import face_locations
from PIL import Image, ImageDraw, ImageFont
import cv2
import pandas as pd
import json
# we are only going to use 4 attributes
COLS = ['Male', 'Asian', 'White', 'Black']
N_UPSCLAE = 1

# load model
model_path = 'face_model.pkl'
clf, labels = None, None
with open(model_path, 'rb') as f:
    clf, labels = pickle.load(f, encoding='latin1')


def extract_features(img_path):
    """Exctract 128 dimensional features
    """
    X_img = face_recognition.load_image_file(img_path)
    locs = face_locations(X_img, number_of_times_to_upsample=N_UPSCLAE)
    if len(locs) == 0:
        return None, None
    face_encodings = face_recognition.face_encodings(
        X_img, known_face_locations=locs)
    return face_encodings, locs


def predict_one_image(img_path):
    """Predict face attributes for all detected faces in one image
    """
    face_encodings, locs = extract_features(img_path)
    if not face_encodings:
        return None, None
    pred = pd.DataFrame(clf.predict_proba(face_encodings), columns=labels)
    pred = pred.loc[:, COLS]
    locs = pd.DataFrame(locs, columns=['top', 'right', 'bottom', 'left'])
    df = pd.concat([pred, locs], axis=1)

    # For Display Image
    # img = draw_attributes(img_path, df)
    # plt.imshow(img)
    # plt.show()

    # Create new column for display result
    df['Gender'] = df.apply(
        lambda row: 'Male' if row['Male'] >= 0.5 else 'Female', axis=1)
    df['Race'] = df[['Asian', 'White', 'Black']].idxmax(axis=1)

    result = df.loc[:, ['Gender', 'Race', 'top',
                        'right', 'bottom', 'left']].to_dict('records')
    return result


def draw_attributes(img_path, df):
    """Write bounding boxes and predicted face attributes on the image
    """
    img = Image.open(img_path)
    img = np.array(img)
    # img  = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    for row in df.iterrows():
        top, right, bottom, left = row[1][4:].astype(int)
        if row[1]['Male'] >= 0.5:
            gender = 'Male'
        else:
            gender = 'Female'

        race = np.argmax(row[1][1: 4])
        text_showed = "{} {}".format(race, gender)

        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        img_width = img.shape[1]
        cv2.putText(img, text_showed, (left + 6, bottom - 6),
                    font, 0.5, (255, 255, 255), 1)
    return img


def main(argv):
    if argv and argv[0] == '--test':
        print('test')
        result = predict_one_image('picture/self.jpg')
        print(result)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
