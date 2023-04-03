#!/usr/bin/python
# -*- coding: utf-8 -*-

# Este NÃO é um programa ROS

from __future__ import print_function, division 

import cv2
import os,sys, os.path
import numpy as np
from time import sleep

from triutil import *

print("Rodando Python versão ", sys.version)
print("OpenCV versão: ", cv2.__version__)
print("Diretório de trabalho: ", os.getcwd())

# Arquivos necessários
video = "triangulos.mp4"

def dentro_fora_triangulo(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_triangulo = cv2.inRange(hsv, (0, 50, 50), (9, 255, 255))
    triangulo = np.where(mask_triangulo==255)
    y_lista, x_lista = triangulo
    x_min, y_min, x_max, y_max = min(x_lista), min(y_lista), max(x_lista), max(y_lista)
    a, b, c = (x_lista[int(np.argmin(y_lista))], y_min), (x_min, y_max), (x_max, y_max)

    mask_estrela = cv2.inRange(hsv, (90, 50, 50), (128, 255, 255))
    contornos_azuis, arvore = cv2.findContours(mask_estrela, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    estrela = sorted(contornos_azuis, key=lambda x: cv2.contourArea(x))[-1]
    centro_estrela = (int(estrela[:,:,0].mean()), int(estrela[:,:,1].mean()))

    text = "Dentro" if point_in_triangle(a, b, c, centro_estrela) else "Fora"
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

    return frame

if __name__ == "__main__":

    # Inicializa a aquisição da webcam
    cap = cv2.VideoCapture(video)


    print("Se a janela com a imagem não aparecer em primeiro plano dê Alt-Tab")

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret == False:
            #print("Codigo de retorno FALSO - problema para capturar o frame")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            #sys.exit(0)

        #TODO: seu código vai aqui

        saida = dentro_fora_triangulo(frame)

        # NOTE que em testes a OpenCV 4.0 requereu frames em BGR para o cv2.imshow
        cv2.imshow('imagem', saida)
        sleep(0.01)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()