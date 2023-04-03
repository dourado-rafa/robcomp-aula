import cv2
import math
import numpy as np

# ========== [ Visualizaçao ] ========== #

def desenha(img, forma, inicio, fim, tamanho=3, cor=(0,0,255)):

    forma(img, inicio, fim, cor, tamanho)

def mira(img, ponto, tamanho=3, cor=(0,0,255)):

    x, y = int(ponto[0]), int(ponto[1])
    cv2.line(img, (x - tamanho, y), (x + tamanho, y), cor, tamanho)
    cv2.line(img, (x, y - tamanho), (x, y + tamanho), cor, tamanho)
    
def desenha_circulos(img, circulos, tamanho=-1, cor=(0,0,255)):

    if circulos is not None:
        for circulo in circulos[0]:
            x, y, raio = circulo
            cv2.circle(img, (int(x), int(y)), int(raio), cor, tamanho)


# ========== [ Processamento de Imagem ] ========== #

CORES_HSV = {'black' : ((0, 0, 0), (180, 255, 30)),
             'white' : ((0, 0, 230), (180, 18, 255)),
             'gray'  : ((0, 0, 40), (180, 18, 230)),

             'red'   : ((159, 50, 50), (180, 255, 255)),
             'green' : ((36, 50, 50), (89, 255, 255)),
             'blue'  : ((90, 50, 50), (128, 255, 255)),
             'yellow': ((25, 50, 50), (35, 255, 255)),
             'purple': ((129, 50, 50), (158, 255, 255)),
             'orange': ((10, 50, 50), (24, 255, 255))}

def segmentador(img, min, max):

    hsv = img.copy()
    mask = cv2.inRange(hsv, (min, 50, 50), (max, 255, 255))

    return mask

def encontrar_contornos(img, tipo=cv2.RETR_EXTERNAL, maior=False, min=1):
    contornos, arvore = cv2.findContours(img, tipo, cv2.CHAIN_APPROX_NONE)

    contornos = [c for c in contornos if c >= min]
    contornos = sorted(contornos, key=lambda c: cv2.contourArea(c))

    if maior:

        return contornos[-1]
    return contornos

def encontrar_centro(contorno, img=None):

    centro = (int(contorno[:,:,0].mean()), int(contorno[:,:,1].mean()))

    if img is not None:
        saida = img.copy()
        mira(saida, centro)

        return centro, saida
    return centro

def encontrar_retangulo(contorno, img=None):

    inicio = (min(contorno[:, :, 0])[0], min(contorno[:, :, 1])[0])
    fim = (max(contorno[:, :, 0])[0], max(contorno[:, :, 1])[0])
    retangulo = (inicio, fim)

    if img is not None:
        saida = img.copy()
        desenha(saida, cv2.rectangle, inicio, fim)

        return retangulo, saida
    return retangulo

#  ========== [ Reconhecimento de Imagens com MobileNet] ========== #

CLASSES_MOBILENET = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
CORES_MOBILENET = np.random.uniform(0, 255, size=(len(CLASSES_MOBILENET), 3))

def inicia_net(diretorio="./"):
    print('[INFO] loading model...')
    mobilenet = cv2.dnn.readNetFromCaffe(f'{diretorio}MobileNetSSD_deploy.prototxt.txt', f'{diretorio}MobileNetSSD_deploy.caffemodel')
    return mobilenet

def detecta(mobilenet, frame, CONFIANCA=70, CORES=CORES_MOBILENET, CLASSES=CLASSES_MOBILENET):

    img = frame.copy()
    l, a = img.shape[:2] # Tamanho da imagem completa
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5) # Extraindo dados da imagem (blob)

    print("[INFO] computing object detections...")
    mobilenet.setInput(blob)
    deteccoes = mobilenet.forward()

    resultados = []

    # Percorre todas as deteccoes
    for i in np.arange(0, deteccoes.shape[2]):
        confianca = deteccoes[0, 0, i, 2]*100 # Confianca associada a predicao

        # Processando predicoes com confiança maior que o minimo estabelicido
        if confianca >= CONFIANCA:
            index = int(deteccoes[0, 0, i, 1]) # Indice da predicao
            cor, classe = CORES[index], CLASSES[index] # Cor e Classe referentes ao indece da predicao

            caixa = deteccoes[0, 0, i, 3:7] * np.array([a, l, a, l]) # Enquadramento da predicao (caixa vem em funcao de porcentagem do tamanho da imagem completa)
            inicioX, inicioY, fimX, fimY = caixa.astype("int")

            # display the prediction
            legenda = f"{classe}: {confianca:.2f}%"
            print(f"[INFO] {legenda}")
            cv2.rectangle(img, (inicioX, inicioY), (fimX, fimY), cor, 2) # Desenha o retangulo na imagem
            y = inicioY - 15 if inicioY - 15 > 15 else inicioY + 15 # posiciona a ledfima em um local legivel
            cv2.putText(img, legenda, (inicioX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2) # Escreve a legenda

            resultados.append({"classe": classe, 
                               "confianca": confianca,
                               "inicio": (inicioX, inicioY),
                               "fim": (fimX, fimY)
                              })

    return img, resultados