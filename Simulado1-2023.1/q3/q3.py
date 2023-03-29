#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Este NÃO é um programa ROS

from __future__ import print_function, division 

import cv2
import os,sys, os.path
import numpy as np

print("Rodando Python versão ", sys.version)
print("OpenCV versão: ", cv2.__version__)
print("Diretório de trabalho: ", os.getcwd())

# Arquivos necessários
video = "dogtraining.mp4"

def desenha_circulos(img, circulos, cor=(0,0,255)):

    if circulos is not None:
        saida = img.copy()
        for circulo in circulos[0]:
            x, y, raio = circulo
            cv2.circle(saida, (int(x), int(y)), int(raio), cor, thickness=-1)

    return saida

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

print('[INFO] loading model...')
mobilenet = cv2.dnn.readNetFromCaffe('./MobileNetSSD_deploy.prototxt.txt', './MobileNetSSD_deploy.caffemodel')
ultima_distancia = 1e100

def detecta_dog(net, frame):
    l, a = frame.shape[:2] # Tamanho da imagem completa
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5) # Extraindo dados da imagem (blob)
    print("[INFO] computing object detections...")
    net.setInput(blob)
    deteccoes = mobilenet.forward()

    # Percorre todas as deteccoes
    for i in np.arange(0, deteccoes.shape[2]):
        confianca = deteccoes[0, 0, i, 2]*100 # Confianca associada a predicao

        # Processando predicoes com confiança maior que o minimo estabelicido
        if confianca > 30:
            index = int(deteccoes[0, 0, i, 1]) # Indice da predicao
            classe = CLASSES[index] # Classe referente ao indece da predicao
            if classe == 'dog':

                dog = deteccoes[0, 0, i, 3:7] * np.array([a, l, a, l]) # Enquadramento da predicao (caixa vem em funcao de porcentagem do tamanho da imagem completa)
                inicioX, inicioY, fimX, fimY = dog.astype("int")
                centro_dog = (int(np.mean([inicioX, fimX])), int(np.mean([inicioY, fimY])))

                # display the prediction
                legenda = f"{classe}: {confianca:.2f}%"
                print(f"[INFO] {legenda}")
                cv2.rectangle(frame, (inicioX, inicioY), (fimX, fimY), (255,0,0), 2) # Desenha o retangulo na imagem
                y = inicioY - 15 if inicioY - 15 > 15 else inicioY + 15 # posiciona a ledfima em um local legivel
                cv2.putText(frame, legenda, (inicioX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2) # Escreve a legenda
                return centro_dog



if __name__ == "__main__":

    # Inicializa a aquisição da webcam
    cap = cv2.VideoCapture(video)

    print("Se a janela com a imagem não aparecer em primeiro plano dê Alt-Tab")

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret == False:
            #print("Codigo de retorno FALSO - problema para capturar o frame")
            #cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            break
    
        # TODO: seu código vai aqui
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_azul = cv2.inRange(hsv, (80, 50, 50), (130, 255, 255))
        bola = cv2.HoughCircles(mask_azul, method=cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=40, param2=12, minRadius=30, maxRadius=70)[0][0]
        centro_bola, raio_bola = (int(bola[0]), int(bola[1])), int(bola[2])
        cv2.circle(frame, centro_bola, raio_bola, (0,0,255), 3)

        centro_dog = detecta_dog(mobilenet, frame)
        if centro_dog is not None:
            distancia = np.sqrt((bola[0]-centro_dog[0])**2 + (bola[1]-centro_dog[1])**2)
            cv2.putText(frame, f'distancia: {distancia:.2f}', (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
            if distancia > ultima_distancia:
                cv2.putText(frame, 'REX, CORRE!', (5,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
            ultima_distancia = float(distancia)

        # NOTE que em testes a OpenCV 4.0 requereu frames em BGR para o cv2.imshow
        cv2.imshow('imagem', frame)

        # Pressione 'q' para interromper o video
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

