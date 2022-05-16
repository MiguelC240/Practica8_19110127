import numpy as np 
import cv2 
from matplotlib import pyplot as plt

Img = cv2.imread("Imagen_Dia.jpg",cv2.COLOR_BGR2GRAY)
esca = cv2.resize(Img,dsize=(300,300))
cv2.imshow('Imagen Dia',esca)

######################## LAPLACIANO ###########################

lap = cv2.Laplacian(esca,cv2.CV_64F)# Detección de borde de Laplace
lap = np.uint8(np.absolute(lap))## Ir al valor absoluto de vuelta
cv2.imshow("Laplaciano",lap)

######################## SOBEL X ###########################

sobelX = cv2.Sobel(esca,cv2.CV_64F,1,0)#x gradiente de dirección
sobelX = np.uint8(np.absolute(sobelX))#x gradiente de dirección valor absoluto
cv2.imshow("Sobel X", sobelX)


######################## SOBEL Y ###########################
 
sobelY = cv2.Sobel(esca,cv2.CV_64F,0,1)#y gradiente de dirección
sobelY = np.uint8(np.absolute(sobelY))#y valor absoluto del gradiente de dirección
cv2.imshow("Sobel Y", sobelY)

######################## SOBEL COMBINADO ###########################

sobelCombined = cv2.bitwise_or(sobelX,sobelY)
cv2.imshow("Sobel Combinado", sobelCombined)

######################## CANNY ###########################

canny = cv2.Canny(esca,30,150)
cv2.imshow("Canny",canny)

cv2.waitKey(0)
cv2.destroyAllWindows()


    
