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
import time
from typing import Dict
# import matplotlib.pyplot as plt
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


def predict_one_image(img_path, ref_position: Dict[str, int] = None):
    """Predict face attributes for all detected faces in one image
    """
    # Find face
    face_encodings, locs = extract_features(img_path)

    # Select face
    if face_encodings and ref_position:
        # select the closest face centroid
        selected = {'index': None, 'distance': None}
        centroid = {
            'x': (ref_position['position_right'] + ref_position['position_left']) / 2,
            'y': (ref_position['position_bottom'] + ref_position['position_top']) / 2}
        for index, loc in enumerate(locs):
            loc_x = (loc[1] + loc[3]) / 2
            loc_y = (loc[2] + loc[0]) / 2
            dist = (centroid['x']-loc_x)**2 + (centroid['y']-loc_y)**2
            if not selected['index']:
                selected['index'] = index
                selected['distance'] = dist
            else:
                if dist < selected['distance']:
                    selected['index'] = index
                    selected['distance'] = dist
        face_encodings = [face_encodings[selected['index']]]
        locs = [locs[selected['index']]]
    elif ref_position:
        # use reference position to force encoding
        locs = [[ref_position['position_top'],
                 ref_position['position_right'],
                 ref_position['position_bottom'],
                 ref_position['position_left']]]
        face_encodings = face_recognition.face_encodings(face_recognition.load_image_file(img_path),
                                                         known_face_locations=locs)
    else:
        # return None result
        return {
            'gender': {
                'type': None,
                'confidence': None
            },
            'race': {
                'type': None,
                'confidence': None
            },
            'position_top': None,
            'position_right': None,
            'position_bottom': None,
            'position_left': None,
            'time': int(time.time())}

    # Predict
    pred = pd.DataFrame(clf.predict_proba(face_encodings), columns=labels)
    pred = pred.loc[:, COLS]  # Get only necessary columns

    # Format
    locs = pd.DataFrame(locs, columns=['top', 'right', 'bottom', 'left'])
    df = pd.concat([pred, locs], axis=1)
    df['Race'] = df[['Asian', 'White', 'Black']].idxmax(axis=1)
    df_list = df[['Male', 'Asian', 'White',
                  'Black', 'Race', 'top', 'right', 'bottom', 'left']].to_dict('records')
    return row_to_dict(df_list[0])


def row_to_dict(row: dict):
    gender_type = 'Male' if row['Male'] >= 0.5 else 'Female'
    gender_confidence = row['Male'] if row['Male'] >= 0.5 else 1 - row['Male']
    race_type = row['Race']
    race_confidence = row[row['Race']]
    return {
        'gender': {
            'type': gender_type,
            'confidence': gender_confidence
        },
        'race': {
            'type': race_type,
            'confidence': race_confidence
        },
        'position_top': row['top'],
        'position_right': row['right'],
        'position_bottom': row['bottom'],
        'position_left': row['left'],
        'time': int(time.time())}


def main(argv):
    if argv and argv[0] == '--test':
        print('test')
        # result = predict_one_image('picture/classmate.jpg', {
        #                            'position_top': 240, 'position_right': 810, 'position_bottom': 290, 'position_left': 760})
        # result = predict_one_image('picture/babe_female.jpg')
        # print(json.dumps(result, indent=2))


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
