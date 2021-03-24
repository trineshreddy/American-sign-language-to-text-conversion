import tensorflow
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions
#
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
# from tensorflow.keras.models import Sequential
from keras.preprocessing.image import img_to_array




from tensorflow.keras.models import load_model
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import operator
import cv2
import sys, os

##loading all models

loaded_model_main        =load_model('model/vgg16/model_vgg16_final.h5')
loaded_model_d_i_r_u_z   =load_model('model/vgg16/model_vgg16_d_i_r_u_z.h5')
loaded_model_g_h_p_q     =load_model('model/vgg16/model_vgg16_g_h_p_q.h5')
loaded_model_k_v         =load_model('model/vgg16/model_vgg16_k_v.h5')
loaded_model_s_m_n_t_a_e =load_model('model/vgg16/model_vgg16_s_m_n_t_a_e.h5')

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

    res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    res = cv2.resize(res, (224, 224))


    res = img_to_array(res)

    res = res.reshape((1, res.shape[0], res.shape[1], res.shape[2]))

    res = preprocess_input(res)

    result_main = loaded_model_main.predict(res)

    prediction = {}
    prediction['blank'] = result_main[0][0]
    index = 1
    for i in ascii_uppercase:
        prediction[i] = result_main[0][index]
        index += 1
        ##main
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    current_symbol = prediction[0][0]

    ###diruz
    if (
            current_symbol == 'D' or current_symbol == 'R' or current_symbol == 'U' or current_symbol == 'I' or current_symbol == 'Z'):
        result_diruz = loaded_model_d_i_r_u_z.predict(res)
        prediction = {}
        prediction['D'] = result_diruz[0][0]
        prediction['I'] = result_diruz[0][1]
        prediction['R'] = result_diruz[0][2]
        prediction['U'] = result_diruz[0][3]
        prediction['Z'] = result_diruz[0][4]
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        current_symbol = prediction[0][0]
    # ghpq
    if (current_symbol == 'G' or current_symbol == 'H' or current_symbol == 'P' or current_symbol == 'Q'):
        result_ghpq = loaded_model_g_h_p_q.predict(res)
        prediction = {}
        prediction['G'] = result_ghpq[0][0]
        prediction['H'] = result_ghpq[0][1]
        prediction['P'] = result_ghpq[0][2]
        prediction['Q'] = result_ghpq[0][3]
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        current_symbol = prediction[0][0]

        ##k_v

    if (current_symbol == 'K' or current_symbol == 'V'):
        result_kv = loaded_model_k_v.predict(res)
        prediction = {}
        prediction['K'] = result_kv[0][0]
        prediction['V'] = result_kv[0][1]
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        current_symbol = prediction[0][0]
    if (current_symbol == 'S' or current_symbol == 'M' or current_symbol == 'N' or current_symbol == 'T' or current_symbol == 'A' or current_symbol == 'E'):
        result_smntae = loaded_model_s_m_n_t_a_e.predict(res)
        prediction = {}
        prediction['A'] = result_smntae[0][0]
        prediction['E'] = result_smntae[0][1]
        prediction['M'] = result_smntae[0][2]
        prediction['N'] = result_smntae[0][3]
        prediction['S'] = result_smntae[0][4]
        prediction['T'] = result_smntae[0][5]
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