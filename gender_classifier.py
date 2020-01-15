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


def predict_one_image(img_path):
    """Predict face attributes for all detected faces in one image
    """
    face_encodings, locs = extract_features(img_path)

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

    pred = pd.DataFrame(clf.predict_proba(face_encodings), columns=labels)
    pred = pred.loc[:, COLS]
    locs = pd.DataFrame(locs, columns=['top', 'right', 'bottom', 'left'])
    df = pd.concat([pred, locs], axis=1)

    # For Display Image
    img = draw_attributes(img_path, df)
    plt.imshow(img)
    plt.show()

    df['Race'] = df[['Asian', 'White', 'Black']].idxmax(axis=1)
    df_list = df[['Male', 'Asian', 'White',
                  'Black', 'Race', 'top', 'right', 'bottom', 'left']].to_dict('records')
    result = list(map(row_to_json, df_list))
    return result


def row_to_json(row):
    gender = 'Male' if row['Male'] >= 0.5 else 'Female'
    gender_confident = row['Male'] if row['Male'] >= 0.5 else 1 - row['Male']
    race = row['Race']
    race_confident = row[row['Race']]
    return {
        'gender': {
            'gender': gender,
            'confident': gender_confident
        },
        'race': {
            'race': race,
            'confident': race_confident
        },
        'top': row['top'],
        'right': row['right'],
        'bottom': row['bottom'],
        'left': row['left']}


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
        test()
        return
        result = predict_one_image('picture/adult.jpeg')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
