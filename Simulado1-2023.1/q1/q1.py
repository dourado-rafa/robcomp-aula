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

def mira(img, ponto, tamanho=3, cor=(0,0,255)):
    x,y = ponto
    x = int(x)
    y = int(y)

    cv2.line(img,(x - tamanho,y),(x + tamanho,y),cor,2)
    cv2.line(img,(x,y - tamanho),(x, y + tamanho),cor,2)

# Arquivos necessários
video = "bandeiras_movie.mp4"

def desenha_circulos(img, circulos, cor=(0,0,255)):

    if circulos is not None:
        saida = img.copy()
        for circulo in circulos[0]:
            x, y, raio = circulo
            cv2.circle(saida, (int(x), int(y)), int(raio), cor, thickness=-1)

    return saida

######
corner_jp = ((10,10),(100,100))
corner_pl = ((5,5),(200,200))

# Responda dentro desta função. 
# Pode criar e chamar novas funções o quanto quiser
def encontra_japao_polonia_devolve_corners(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask_not_black= cv2.inRange(gray, (10), (255))
    contornos_bandeiras, arvore = cv2.findContours(mask_not_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(hsv, (155, 50, 50), (185, 255, 255))
    contornos_vermelhos, arvore = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Encontra Polonia
    vemelho_polonia = sorted(contornos_vermelhos, key=lambda x: cv2.contourArea(x))[-1] # seleciona o maior contorno
    centro_polonia = (int(vemelho_polonia[:,:,0].mean()), int(vemelho_polonia[:,:,1].mean()))

    # Encontra Japao
    circulo = cv2.HoughCircles(mask_red, method=cv2.HOUGH_GRADIENT, dp=1, minDist=30, param1=40, param2=12, minRadius=10, maxRadius=50)[0]
    centro_japao = (int(circulo[0][0]), int(circulo[0][1]))
    
    # seleciona as bandeira
    for bandeira in contornos_bandeiras:
        if cv2.pointPolygonTest(bandeira, centro_japao, False) == 1:
            jp = bandeira
            cv2.drawContours(frame, [jp], -1, (255,0,0), 3)
        
        elif cv2.pointPolygonTest(bandeira, centro_polonia, False) == 1:
            pl = bandeira
            cv2.drawContours(frame, [pl], -1, (0,255,0), 3)


    corner_jp = ((min(jp[:, :, 0])[0], min(jp[:, :, 1])[0]), (max(jp[:, :, 0])[0], max(jp[:, :, 1])[0]))
    corner_pl = ((min(pl[:, :, 0])[0], min(pl[:, :, 1])[0]), (max(pl[:, :, 0])[0], max(pl[:, :, 1])[0]))

    cv2.putText(frame, 'Japao', np.array(corner_jp[0])-np.array([0,10]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
    cv2.putText(frame, 'Polonia', np.array(corner_pl[0])-np.array([0,10]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))

    

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


