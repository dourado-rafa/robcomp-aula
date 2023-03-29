#!/usr/bin/python
# -*- coding: utf-8 -*-

# Este NÃO é um programa ROS

import time

import cv2
import os,sys, os.path
import numpy as np

import triutil

print("Rodando Python versão ", sys.version)
print("OpenCV versão: ", cv2.__version__)
print("Diretório de trabalho: ", os.getcwd())

# Arquivos necessários
video = "triangulos.mp4"


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

        # Our operations on the frame come here
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ACHAR O TRIANGULO VERDE
        verde1 = np.zeros_like(gray)
        verde1[rgb[:,:,1] > 200] = 255

        contornos, _ = cv2.findContours(verde1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x_list = contornos[0][:,:,0]
        y_list = contornos[0][:,:,1]
        
        baixo_esquerdo = (min(x_list)[0], max(y_list)[0])
        baixo_direito = (max(x_list)[0], max(y_list)[0])

        idx_y_menor = np.argmin(y_list)
        topo = (x_list[idx_y_menor][0], y_list[idx_y_menor][0])

        print(topo, baixo_direito, baixo_esquerdo)

        # ACHAR ESTRELA
        azul = np.zeros_like(gray)
        azul[rgb[:,:,2] > 200] = 255

        contornos_azul, _ = cv2.findContours(azul, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y = np.mean(contornos_azul[0][:,:,0]), np.mean(contornos_azul[0][:,:,1])

        text = 'FORA'
        if triutil.point_in_triangle(topo, baixo_direito, baixo_esquerdo, (x, y)):
            text = 'DENTRO'


        cv2.putText(frame,
            text, (0, 20), cv2.FONT_ITALIC, 1, (255, 255, 255), 4)


        # NOTE que em testes a OpenCV 4.0 requereu frames em BGR para o cv2.imshow
        cv2.imshow('imagem', frame)

        time.sleep(0.05)        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


