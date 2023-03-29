#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Este NÃO é um programa ROS

import math
import cv2
import os,sys, os.path
import numpy as np

print("Rodando Python versão ", sys.version)
print("OpenCV versão: ", cv2.__version__)
print("Diretório de trabalho: ", os.getcwd())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def detect(frame):
    image = frame.copy()
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    results = []

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence


        if confidence > 0.5:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            results.append((CLASSES[idx], confidence*100, (startX, startY),(endX, endY) ))
        
    return results


# Arquivos necessários
video = "dogtraining.mp4"

if __name__ == "__main__":

    # Inicializa a aquisição da webcam
    cap = cv2.VideoCapture(video)

    net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')


    print("Se a janela com a imagem não aparecer em primeiro plano dê Alt-Tab")

    ultima_distancia = 1000000000

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret == False:
            #print("Codigo de retorno FALSO - problema para capturar o frame")
            #cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            break
            

        # Our operations on the frame come here
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # BOLA
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        azul_menor, azul_maior = 110, 130
        azul_com_bola = cv2.inRange(hsv, (azul_menor, 50, 50), (azul_maior, 255, 255))
        azul_com_bola = cv2.morphologyEx(azul_com_bola, cv2.MORPH_CLOSE, np.ones((11, 11)))
        brancos_y, brancos_x = np.where(azul_com_bola == 255)
        x_centro, y_centro = (np.mean(brancos_x), np.mean(brancos_y))
        print(x_centro, y_centro)
        cv2.circle(frame, (int(x_centro), int(y_centro)), 5, (255, 255, 255), -1)

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,20,
                            param1=50,param2=30,minRadius=30,maxRadius=55)

        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            centro = i[0], i[1]
            if math.sqrt((x_centro - i[0]) ** 2 + (y_centro - i[1]) ** 2) < 10:
                raio = i[2]

        # CACHORRO
        res = detect(frame)
        if len(res) > 0:
            cachorro = res[0]
            x_cachorro = (res[0][2][0] + res[0][3][0]) / 2
            y_cachorro = (res[0][2][1] + res[0][3][1]) / 2

            distancia = math.sqrt((x_centro - x_cachorro) ** 2 + (y_centro - y_cachorro) ** 2)
            cv2.putText(frame, 
                        f'Distancia: {distancia:.2f}',
                        (0, 20), cv2.FONT_ITALIC,
                        1, (255, 255, 255), 5)
            
            if distancia > ultima_distancia:
                print('CORRE')
                cv2.putText(frame, 
                        'CORRE',
                        (0, 50), cv2.FONT_ITALIC,
                        1, (255, 255, 255), 5)
            
            ultima_distancia = distancia
        

        # NOTE que em testes a OpenCV 4.0 requereu frames em BGR para o cv2.imshow
        cv2.imshow('imagem', frame)

        # Pressione 'q' para interromper o video
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

