import cv2 as cv
import numpy as np

def generateCoord(img, paramLineales):
    pendiente, inter = paramLineales
    y1 = img.shape[0]
    y2 = int(y1*(5/8))
    x1 = int((y1 - inter)/pendiente)
    x2 = int((y2 - inter)/pendiente)
    return np.array([x1,y1,x2,y2])

def showLines(img, lineas):
    fitIzq = []
    fitDer = []
    for linea in lineas:
        x1,y1,x2,y2 = linea.reshape(4)
        paramPolLineal = np.polyfit((x1,x2), (y1,y2), 1)
        pendiente = paramPolLineal[0]
        inter = paramPolLineal[1]
        if pendiente < 0:
            fitIzq.append((pendiente, inter))
        else:
            fitDer.append((pendiente,inter))
    prom_izq = np.average(fitIzq, axis = 0)
    prom_der = np.average(fitDer, axis=0)
    lineaIzq = generateCoord(img, prom_izq)
    lineaDer = generateCoord(img, prom_der)
    return np.array([lineaIzq, lineaDer])
def processImg(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (7,7),0)
    img = cv.Canny(img, 130, 170)
    return img

def limitRegion(img):
    h = img.shape[0]
    tri = np.array([[(200, h), (900, h), (590, 240)]])
    mask = np.zeros_like(img)
    cv.fillPoly(mask, tri, 255)
    masked_image = cv.bitwise_and(img, mask)
    return masked_image

def houghLines(img, lineas):
    img_lineas = np.zeros_like(img)
    if lineas is not None:
        for linea in lineas:
            x1,y1,x2,y2 = linea.reshape(4)
            cv.line(img_lineas, (x1,y1), (x2,y2), (0,0,255), 10)
    return img_lineas

cap = cv.VideoCapture("Videos/test_countryroad.mp4")

while(cap.isOpened()):
    _,frame = cap.read()
    frame = cv.resize(frame, (np.array(frame).shape[1] // 2, np.array(frame).shape[0] // 2))
    filteredImg = processImg(frame)
    cropped = limitRegion(filteredImg)
    lineas = cv.HoughLinesP(cropped, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=2)
    linea_prom = showLines(frame, lineas)
    img_lineas = houghLines(frame, linea_prom)
    mix = cv.addWeighted(frame, 0.7, img_lineas, 1, 1)
    cv.imshow("asd", mix)
    cv.waitKey(1)
if cv.waitKey(20) & 0xFF == "w":
    cv.destroyAllWindows()


