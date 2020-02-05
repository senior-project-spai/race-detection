import matplotlib.pyplot as plt
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
N_UPSCLAE = 2

# load model
model_path = 'face_model.pkl'
clf, labels = None, None
with open(model_path, 'rb') as f:
    clf, labels = pickle.load(f, encoding='latin1')

# ------------------------------------ AGE ----------------------------------- #

age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
            '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
# ---------------------------------------------------------------------------- #


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

    ''' TODO: AGE
    if not face_encodings:
        return []

# ------------------------------------ AGE ----------------------------------- #

    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    for (top, right, bottom, left) in locs:
        face_img = image[top:bottom, left:right].copy()
        blob = cv2.dnn.blobFromImage(
            face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        print(age_preds)
        print("Age Range: {}".format(age))

# ---------------------------------------------------------------------------- #
    '''
    # Select face
    if face_encodings and ref_position:
        # select the closest face centroid
        print("INFO: use face_encodings and ref_position")
        selected = {'index': None, 'distance': None}
        centroid = {
            'x': (ref_position['position_right'] + ref_position['position_left']) / 2,
            'y': (ref_position['position_bottom'] + ref_position['position_top']) / 2}
        for index, loc in enumerate(locs):
            loc_x = (loc[1] + loc[3]) / 2
            loc_y = (loc[2] + loc[0]) / 2
            dist = (centroid['x']-loc_x)**2 + (centroid['y']-loc_y)**2
            if selected['index'] == None:
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
        print("INFO: use only ref_position")
        locs = [[ref_position['position_top'],
                 ref_position['position_right'],
                 ref_position['position_bottom'],
                 ref_position['position_left']]]
        face_encodings = face_recognition.face_encodings(face_recognition.load_image_file(img_path),
                                                         known_face_locations=locs)
    elif face_encodings:
        print("INFO: use only face_encodings")
        face_encodings = face_encodings[0:1]
        locs = locs[0:1]
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
    print("INFO: Predict Location: {}".format(locs))
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

# ----------------------------------- Test ----------------------------------- #


def test():
    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        locs = face_locations(frame, number_of_times_to_upsample=N_UPSCLAE)

        if locs:
            # ------------------------------------ AGE ----------------------------------- #

            for (top, right, bottom, left) in locs:
                face_img = frame[top:bottom, left:right].copy()
                blob = cv2.dnn.blobFromImage(
                    face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = age_list[age_preds[0].argmax()]
                print(age_preds)
                print("Age Range: {}".format(age))
                cv2.rectangle(frame, (left, top),
                              (right, bottom), (0, 0, 255), 2)
# ---------------------------------------------------------------------------- #

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
# ---------------------------------------------------------------------------- #


def main(argv):
    if argv and argv[0] == '--test':
        print('test')
        # result = predict_one_image('picture/classmate.jpg', {
        #                            'position_top': 240, 'position_right': 810, 'position_bottom': 290, 'position_left': 760})
        result = predict_one_image('picture/babe_female.jpg', {
                                   'position_top': 136, 'position_right': 681, 'position_bottom': 394, 'position_left': 423})
        # result = predict_one_image('picture/babe_female.jpg')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
