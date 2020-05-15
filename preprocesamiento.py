import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#Hola

# Funciones
def changesize(image):
    [fil, col, cap] = image.shape
    fil_new = int(fil / 3)
    col_new = int(col / 3)
    image = cv.resize(image, (col_new, fil_new))
    return image


def EliminarRuido(image):
        plt.hist(image.ravel(), bins=256)
        #cv.imshow("orignial", image)


def Binalizacion(image):
    ret, thresh = cv.threshold(image, 0, 255,
                               cv.THRESH_BINARY_INV + cv.THRESH_OTSU)  # Busca umbral para separar fondo de texto
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=1)  # Rellenar huecos
    thresh = 255 - thresh  # Fonde blanco, letras negras
    return thresh


def Contorno(image):
    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours[:], len(contours)


def Formato_MNIS(image):
    image = cv.resize(image, (28, 28))
    return image.ravel()


def Coordenadas(tam, contorno):
    posiciones = np.zeros((size, 4))  # Arreglo donde se guardará las cordenas x,y,w,h para realizar los cortes
    for i in range(0, tam):
        posiciones[i] = cv.boundingRect(contorno[i])  # encontrar x, y , w, h

    posiciones = sorted(posiciones, key=lambda posiciones: posiciones[0])  # Ordena los objetos
    return posiciones


# Varibles
kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
k = -1
d = 0

# Inicio

# Captura, cambio de tamaño, y imagen en escala de grises

img = cv.imread("grande.jpeg")
img = changesize(img)
img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#EliminarRuido(img_grey)

# Binalizacion
thresh = Binalizacion(img_grey)

# Find Contours
objs, size = Contorno(thresh)

data = np.zeros((size, 784))  # Arreglo para guardar los datos en la base de datos MNIST

coordenadas = Coordenadas(size, objs)



# Ciclo para recortar de la imagen orginal, cada uno de los objetos: xp->posición presente, xf->posicion futura
for j in range(size-1):
    k += 1
    xp = int(coordenadas[k][0])
    yp = int(coordenadas[k][1])
    wp = int(coordenadas[k][2])
    hp = int(coordenadas[k][3])

    if k < size - 1:
        xf = int(coordenadas[k + 1][0])
    else:
        xf = yp + 50

    xdif = np.abs(xp - xf)
    print(xdif)
    if xdif < 30:  # Comparar si dos regiones pertenece a la misma imagen

        yf = int(coordenadas[k + 1][1])
        wf = int(coordenadas[k + 1][2])
        hf = int(coordenadas[k + 1][3])

        ydif = np.abs(yp - yf)
        xp = min(xf, xp)
        yp = min(yp, yf)
        wp = max(wf, wp)
        hp = ydif + hf
        d += 1
        k += 1

    img2 = thresh[yp:yp + hp, xp:xp + wp]  # Recorte de la imagen
    cv.imshow("img", img2)
    img2 = Formato_MNIS(img2)  # Redimensión de un vector de 784
    data[j] = img2
    num_objetos = k + d
    if num_objetos >= size:
        break

    cv.waitKey(0)

# show
plt.show()
cv.waitKey()
cv.destroyAllWindows()