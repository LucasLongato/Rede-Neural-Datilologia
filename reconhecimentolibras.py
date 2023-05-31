import cv2
import mediapipe as mp
import pandas as pd  
import os
import numpy as np 
import pickle
import time

def image_processed(hand_img):

    # Processamento de imagem
    # 1. Converta BGR para RGB
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

   # 2. Vire a img no eixo Y
    img_flip = cv2.flip(img_rgb, 1)

    mp_hands = mp.solutions.hands

    # Inicializar o modulo de  Mãos
    hands = mp_hands.Hands(static_image_mode=True,
    max_num_hands=1, min_detection_confidence=0.7)

    #Resultados
    output = hands.process(img_flip)

    hands.close()

    try:
        data = output.multi_hand_landmarks[0]
        #print(data)
        data = str(data)

        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_garbage = []

        for i in data:
            if i not in garbage:
                without_garbage.append(i)
                        
        clean = []

        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])
        return(clean)
    except:
        return(np.zeros([1,63], dtype=int)[0])



with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)



cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
i = 0    
while True:
    
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
  
    data = image_processed(frame)
    
    
    data = np.array(data)
    y_pred = svm.predict(data.reshape(-1,63))
    print(y_pred)
   
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    
    org = (50, 100)
    
   
    fontScale = 3
    
    
    color = (255, 0, 0)
    
    # Espessura da linha de 2 px
    thickness = 5
    
    # Usando o método cv2.putText()
    frame = cv2.putText(frame, str(y_pred[0]), org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()