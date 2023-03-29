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
video = "bandeiras_movie.mp4"

######
corner_jp = ((10,10),(100,100))
corner_pl = ((5,5),(200,200))

# Responda dentro desta função. 
# Pode criar e chamar novas funções o quanto quiser
def encontra_japao_polonia_devolve_corners(bgr):
    frame = bgr.copy()

    # JAPAO
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mascara_brancos = np.zeros_like(gray)
    mascara_brancos[gray > 200] = 255
    contornos, _ = cv2.findContours(mascara_brancos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    japao_contorno = contornos[0]
    for ct in contornos:
        if cv2.contourArea(ct) > cv2.contourArea(japao_contorno):
            japao_contorno = ct
    
    x_list = japao_contorno[:,:,0]
    y_list = japao_contorno[:,:,1]
    corner_jp = x, y, x2, y2 = min(x_list)[0], min(y_list)[0], max(x_list)[0], max(y_list)[0]
    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 5)
    cv2.putText(frame,
                "Japao",
                (x, y2 + 20),
                cv2.FONT_ITALIC,
                1, (0, 255, 0), 2)

    # POLONIA
    mascara_bandeiras = np.zeros_like(gray)
    mascara_bandeiras[gray > 20] = 255
    contornos2, _ = cv2.findContours(mascara_bandeiras, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    filtro_vermelho1 = np.zeros_like(gray)
    filtro_vermelho1[bgr[:,:,2] > 200] = 255

    filtro_vermelho2 = np.zeros_like(gray)
    filtro_vermelho2[bgr[:,:,1] < 200] = 255

    filtro_vermelho = cv2.bitwise_and(filtro_vermelho1, filtro_vermelho2)
    cv2.imshow("vermelho", filtro_vermelho)

    for i, ct in enumerate(contornos2):
        uma_bandeira = np.zeros_like(gray)
        cv2.drawContours(uma_bandeira, [ct], -1, 255, -1)

        brancos_um_bandeira = cv2.bitwise_and(uma_bandeira, filtro_vermelho)
        brancos_dentro = np.sum(brancos_um_bandeira / 255)

        area = cv2.contourArea(ct)
        if brancos_dentro / area >= 0.4 and brancos_dentro / area <= 0.6:
            polonia_contorno = ct
            break
    
    x_list = polonia_contorno[:,:,0]
    y_list = polonia_contorno[:,:,1]
    corner_pl = x, y, x2, y2 = min(x_list)[0], min(y_list)[0], max(x_list)[0], max(y_list)[0]
    cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 5)
    cv2.putText(frame, "Polonia", (x, y2+20), 
                cv2.FONT_ITALIC, 1, (255, 0, 0),
                2)

    return frame, corner_jp, corner_pl

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

        # Programe só na função encontra_japao_devolve_corners. Pode criar funções se quiser
        saida, japao, polonia = encontra_japao_polonia_devolve_corners(frame)

        print("Corners x-y Japao")
        print(japao)
        print("Corners x-y Polonia")
        print(polonia)

        # NOTE que em testes a OpenCV 4.0 requereu frames em BGR para o cv2.imshow
        cv2.imshow('imagem', saida)

        if cv2.waitKey(15) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


