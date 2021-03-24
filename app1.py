import tensorflow
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model,model_from_json
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions

from tensorflow.keras.preprocessing import image


from keras.preprocessing.image import img_to_array




from tensorflow.keras.models import load_model

import operator
import cv2



##loading all models
##loading general model
json_file = open("model/Model1/model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
loaded_model.load_weights("model/Model1/model-bw.h5")
##loading dru model
json_file = open("model/Model1/model-bw_dru.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model_dru = model_from_json(model_json)
loaded_model_dru.load_weights("model/Model1/model-bw_dru.h5")
####loading smn model
json_file = open("model/Model1/model-bw_smn.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model_smn = model_from_json(model_json)
loaded_model_smn.load_weights("model/Model1/model-bw_smn.h5")
##loading tkdi model
json_file = open("model/Model1/model-bw_tkdi.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model_tkdi = model_from_json(model_json)
loaded_model_tkdi.load_weights("model/Model1/model-bw_tkdi.h5")


ascii_uppercase=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    frame = cv2.flip(frame, 1)

    x1 = int(0.5 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(0.5 * frame.shape[1])

    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)

    roi = frame[y1:y2, x1:x2]

    minValue = 70

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)

    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    test_image = cv2.resize(res, (128, 128))
    result = loaded_model.predict(test_image.reshape(1, 128, 128, 1))

    prediction = {}
    prediction['blank'] = result[0][0]
    index = 1
    for i in ascii_uppercase:
        prediction[i] = result[0][index]
        index += 1
        # main_prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    current_symbol = prediction[0][0]

    ###dru
    if (current_symbol == 'D' or current_symbol == 'R' or current_symbol == 'U'):
        result_dru = loaded_model_dru.predict(test_image.reshape(1, 128, 128, 1))
        prediction = {}
        prediction['D'] = result_dru[0][0]
        prediction['R'] = result_dru[0][1]
        prediction['U'] = result_dru[0][2]
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        current_symbol = prediction[0][0]
    ##dtiu
    if (current_symbol == 'D' or current_symbol == 'I' or current_symbol == 'K' or current_symbol == 'T'):
        result_tkdi = loaded_model_tkdi.predict(test_image.reshape(1, 128, 128, 1))
        prediction = {}
        prediction['D'] = result_tkdi[0][0]
        prediction['I'] = result_tkdi[0][1]
        prediction['K'] = result_tkdi[0][2]
        prediction['T'] = result_tkdi[0][3]
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        current_symbol = prediction[0][0]

        ##mns

    if (current_symbol == 'M' or current_symbol == 'N' or current_symbol == 'S'):
        result_smn = loaded_model_smn.predict(test_image.reshape(1, 128, 128, 1))
        prediction = {}
        prediction['M'] = result_smn[0][0]
        prediction['N'] = result_smn[0][1]
        prediction['S'] = result_smn[0][2]
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

        current_symbol = prediction[0][0]

    print(current_symbol)

    cv2.putText(frame, current_symbol, (10, 120), cv2.FONT_HERSHEY_PLAIN, 10, (0, 255, 255), 10)
    cv2.imshow("Frame", frame)
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()