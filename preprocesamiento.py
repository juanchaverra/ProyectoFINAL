import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# Funciones
def changesize(image):
    [fil, col, cap] = image.shape
    fil_new = int(fil / 3)
    col_new = int(col / 3)
    image = cv.resize(image, (col_new, fil_new))
    return image


# Varibles
kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
mser = cv.MSER_create()
yf = 0
k = -1

# Inicio

# Captura, cambio de tamaño, y imagen en escala de grises
img = cv.imread("capturafinal2.jpeg")
img = changesize(img)
cv.imshow("Original/3", img)
img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
vis = img.copy()

# Binalizacion
ret, thresh = cv.threshold(img_grey, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=1)

# Find Contours
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
objs = contours[:]
size = len(contours)
Coordenadas = np.zeros((size, 4))
thresh =255 - thresh

# Ciclo para obtener las coordenas de los objetos
for i in range(0, size):
    Coordenadas[i] = cv.boundingRect(objs[i])  # x, y , w, h

Coordenadas = sorted(Coordenadas, key=lambda Coordenadas: Coordenadas[0])  # Ordena los objetos en su orden
print(Coordenadas)


# Ciclo para recortar de la imagen orginal, cada uno de los objetos
for j in range(0, size-1):
    k = k + 1
    xp = int(Coordenadas[k][0])
    yp = int(Coordenadas[k][1])
    wp = int(Coordenadas[k][2])
    hp = int(Coordenadas[k][3])

    if k < size-1:
        xf = int(Coordenadas[k + 1][0])
    else:
        xf = yp + 12

    xdif = np.abs(xp - xf)

    if xdif < 10:
        yf = int(Coordenadas[k + 1][1])
        wf = int(Coordenadas[k + 1][2])
        hf = int(Coordenadas[k + 1][3])

        ydif = np.abs(yp -yf)
        xp = min(xf, xp)
        yp = min(yp, yf)
        wp = max(wf, wp)
        hp = ydif + hf

        k += 1

    print(k)
    img2 = thresh[yp:yp + hp, xp:xp + wp]

    cv.imshow("img2", img2)

    cv.waitKey(0)
    if k > size-2:
        break


# show
cv.imshow("Original", img)
cv.imshow("binalizacion", thresh)
plt.show()
cv.waitKey()
cv.destroyAllWindows()
