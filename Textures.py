# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:13:58 2015

@author: klek
"""


from cv2 import *
import cv2

import numpy
import math as m

class Point2:
   def __init__(self, x, y):
       self.x = x
       self.y = y            

class Point3:
    def __init__(self, x, y,z):
       self.x = x
       self.y = y 
       self.z = z
       
def sortPoints(cloud, x, y, z):#сортировка точек в трёхмерном пространстве,
                                #неиспользуется (не помню для чего она была нужна)
    for i in range(len(cloud)):
        l1 = m.sqrt((x-cloud[i].x)*(x-cloud[i].x)+
                    (y-cloud[i].y)*(y-cloud[i].y)+
                    (z-cloud[i].z)*(z-cloud[i].z))
        for j in range(len(cloud)):
            l2 = m.sqrt((x-cloud[j].x)*(x-cloud[j].x)+
                        (y-cloud[j].y)*(y-cloud[j].y)+
                        (z-cloud[j].z)*(z-cloud[j].z))
            if l1 > l2:
                temp = cloud[i]
                cloud[i]=cloud[j]
                cloud[j]=temp
    return cloud
       
   

def getCorners(inputFile):#ищет углы объекта на фотографии, основу взял из интернета
                            #возвращает список точек
    
    foundPoints =[] 
    
    imcolor = cv.LoadImage(inputFile)
    image = cv.LoadImage(inputFile,cv.CV_LOAD_IMAGE_GRAYSCALE)
    cornerMap = cv.CreateMat(image.height, image.width, cv.CV_32FC1)
    # детектор углов Харриса
    cv.CornerHarris(image,cornerMap,3)
    
    for y in range(0, image.height):
        for x in range(0, image.width):
            harris = cv.Get2D(cornerMap, y, x) 
            if harris[0] > 10e-06:
                #cv.Circle(imcolor,(x,y),10,cv.RGB(155, 0, 25))
                point = Point2(x,y)
                foundPoints.append(point)
                #print('x =', x, ' y =', y) 
                
                #cv.NamedWindow('Harris', cv.CV_WINDOW_AUTOSIZE)
                #cv.ShowImage('Harris', imcolor) # show the image
                #print foundPoints
    
    #инициируем массива флагов того, что точка уже учтена в результате
    used = []
    for i in range(len(foundPoints)):
        used.append(0)
    result = []
        
    for i in range(len(foundPoints)):
        points = []
            #усреднение точек
        for j in range(len(foundPoints)):
            l1 = foundPoints[i].x-foundPoints[j].x
            l2 = foundPoints[i].y-foundPoints[j].y
            if (m.fabs(l1) <= 5) and (m.fabs(l2) <= 5) and (used[j]==0):
                points.append(foundPoints[j])
                used[j]=1
        sx = 0
        sy = 0
        #запоминаем точку
        if len(points)!=0:
            for k in range(0,len(points)):
                sx = sx+points[k].x
                sy = sy+points[k].y
            p = Point2(sx/len(points),sy/len(points))
            result.append(p)
    return result
                            
def rotateCloud(cloud,alfa, betta, gamma):#аффинный поворот облака точек
    #матрицы поворота
    Mx = numpy.array([[1,0,0],
                      [0,m.cos(alfa),-m.sin(alfa)],
                       [0, m.sin(alfa),-m.cos(alfa)]])
    My = numpy.array([[m.cos(betta),0, m.sin(betta)],
                       [0,1,0],
                       [-m.sin(betta),0,m.cos(betta)]])
     
    Mz =  numpy.array([[m.cos(gamma), -m.sin(gamma), 0],
                       [m.sin(gamma),m.cos(gamma),0],
                       [0,0,1]])
                   
    
                     
    M = [] 
                     
    for i in range(len(cloud)):
        p = numpy.array([cloud[i].x,cloud[i].y,cloud[i].z])
    
        M.append(numpy.dot(numpy.dot(numpy.dot(Mx,My),Mz), p))#умножение матриц
    
    result = []   
    #возврат результата в виде массива точек
    for i in range(len(M)):
        p3 = Point3(M[i][0],M[i][1],M[i][2])
        result.append(p3)
    return(result)       



def midle(trngl):#середина треугольника (среднее арифметическое точек)
    mx = (trngl[0].x+trngl[1].x+trngl[2].x)/3
    my = (trngl[0].y+trngl[1].y+trngl[2].y)/3
    p = Point2(mx,my)
    return p
    

def copyArray(arr):#копирование массива точек
    result = []
    for i in range(len(arr)):
        el = arr[i]
        result.append(el)
    return result
    
def getIdention(trngl, ident):#смещение в треугольнике, чтобы погрешность при поиске
                                #углов и сторон не влияло на результат (используется в getMeshes)
    result = [] 
    mdl = midle(trngl)
    result = []
    for i in range(3):
        fi = m.atan(m.fabs(mdl.y-trngl[i].y)/m.fabs(mdl.x-trngl[i].x))
        dx = ident*m.cos(fi)
        dy = ident*m.sin(fi)
        if trngl[i].x > mdl.x:
            dx=-dx
        if trngl[i].y > mdl.y:
            dy = -dy
        x=int(trngl[i].x+dx)
        y=int(trngl[i].y+dy)
        p = Point2(x,y)
        result.append(p)
        
    return result

def triangelsEqual(t1,t2): #проверяет один и тотже треугольник или нет
    result = True
    for i in range(3):
        if not((t1[i].x == t2[0].x and t1[i].y == t2[0].y)or(t1[i].x == t2[1].x and t1[i].y == t2[1].y)or(t1[i].x == t2[2].x and t1[i].y == t2[2].y)):
            result = False
    return result

def used(triangles, trngl): #входит ли треугольник в коллекцию
    result = False
    if len(triangles)==0:
        return False
    for i in range(len(triangles)):
        if triangelsEqual(trngl, triangles[i]):
            result = True
    return result
                    
        
def getTriangles(points):#получить все треугольники по точкам
    result = []
    trngl =[]
    for i in range(len(points)):
        for j in range(len(points)):
            if j!=i:
                for k in range(len(points)):
                    if j!=k and i!=k:
                        trngl = []
                        trngl.append(points[i])
                        trngl.append(points[j])
                        trngl.append(points[k])
                        if not used(result, trngl):
                            result.append(trngl)
    return result
    
def pointInTriangle(p, t):#точка в треугольнике или нет
    a = (t[0].x - p.x)*(t[1].y - t[0].y) - (t[1].x-t[0].x)*(t[0].y-p.y)
    b = (t[1].x - p.x)*(t[2].y-t[1].y)-(t[2].x-t[1].x)*(t[1].y-p.y)
    c = (t[2].x-p.x)*(t[0].y-t[2].y)-(t[0].x-t[2].x)*(t[2].y-p.y)
    
    if a*b >= 0 and b*c >= 0:
        return True
    return False
                   


def getMeshes(inputFile, corners, k, iden):#возвращает те треугольники, которые 
                                            #нужно заполнять
    print 'getMeshes start'
    image = cv2.imread(inputFile)
    trngls = getTriangles(corners)
    use = []
    #инициализация массива маркеров использования итого треугольника
    #+ границы поиска
    minX=image.shape[0]-1
    maxX=0
    minY=image.shape[1]-1
    maxY=0
    for i in range(len(trngls)):
        use.append(True)
        
        for j in range(3):
            if trngls[i][j].x > maxX:
                maxX = trngls[i][j].x
            if trngls[i][j].x < minX:
                minX = trngls[i][j].x
            if trngls[i][j].y > maxY:
                maxY = trngls[i][j].y
            if trngls[i][j].y < minY:
                minY = trngls[i][j].y
   
    #поиск граней объекта через лапласиан              
    kernel_size = 1
    scale = 0.09
    delta = 0
    ddepth = cv2.CV_16S
    img = cv2.imread('cube.jpg')
    img = cv2.GaussianBlur(img,(3,3),0)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_lap = cv2.Laplacian(gray,ddepth,ksize = kernel_size,scale = scale,delta = delta)
    
    for x in range(minX, maxX):
        for y in range(minY, maxY):
            for i in range(len(trngls)):
                print x, '//', y, '//', i#вывод для того, чтобы знать что функция выполняется и ничего не повисло
                trngl = getIdention(trngls[i], iden)
                if gray_lap[x][y] != 0:#если пиксель был помечен лапласианом, т.е.
                                        #по нему проходит грань...
                    p = Point2(y,x)  
                    if pointInTriangle(p, trngl):#и лежит внутри треугольника, то
                        use[i]=False             #треугольник не испльзуется
                        
    result = []
    
    for i in range(len(trngls)):
        if use[i]:
            result.append(trngls[i])
           
    return result



def resizeProectiion(pr, crn):#аффинные преобразования для сопоставления двух наборов точек
    left = pr[0]
    right = pr[0]
    bottom = pr[0]

    for i in range(len(pr)):
    #p = pr[i]
        if pr[i].x < right.x:
            right = pr[i]
        if pr[i].x >= left.x:
            left = pr[i]
        if pr[i].y > bottom.y:
            bottom = pr[i]

    rx1 = m.fabs(right.x-left.x)
    ry1 = m.fabs(bottom.y-left.y)
    
    left2 = crn[0]
    right2 = crn[0]
    bottom2 = crn[0]

    for i in range(len(crn)):
    #p = pr[i]
        if crn[i].x < right2.x:
            right2 = cr[i]
        if crn[i].x > left2.x:
            left2 = crn[i]
        if crn[i].y > bottom2.y:
            bottom2 = crn[i]
        
    rx2 = m.fabs(right2.x-left2.x)
    ry2 = m.fabs(bottom2.y-left2.y)
    
    kx = rx2/rx1
    ky = ry2/ry1
    
    dx = left2.x-left.x
    dy = left2.y-left.y
    
    for i in range(len(pr)):
        pr[i].x=pr[i].x+dx
        pr[i].y=pr[i].y+dy
    for i in range(len(pr)):
        pr[i].x = left.x + kx*(pr[i].x-left.x)
        pr[i].y = left.y + ky*(pr[i].y-left.y)
        
    return pr
  
def copy(a):#ещё одно копирование, когда писал эту функцию, забыл про аналог выше...
    result = []
    for i in range(len(a)):
        x = a[i].x
        y = a[i].y
        p = Point2(x,y)
        result.append(p)
    return result

def getConformity(pr, crn):#сопоставление углов объекта на фотографии точкам из облака
    pr1 = copy(pr)
    pr2 = resizeProectiion(pr1, crn)
    
    result = []
    use = []
    for i in range(len(pr2)):
        use.append(False)
    
    for i in range(len(crn)):#сопоставляются те, которые лежат на минимальном расстоянии
        neibor = 0
        l1 = m.sqrt((pr2[0].x-crn[i].x)*(pr2[0].x-crn[i].x)+(pr2[0].y-crn[i].y)*(pr2[0].y-crn[i].y))
        for j in range(len(pr2)):
            l2 = m.sqrt((pr2[j].x-crn[i].x)*(pr2[j].x-crn[i].x)+(pr2[j].y-crn[i].y)*(pr2[j].y-crn[i].y))
            if (l2<l1) and not use[j]:
                l1=l2
                neibor = j
        result.append(pr[neibor])
    return result


def draw(photo_file, cloud, alfa, betta, gamma):#Функция, собирающая всё вместе
    corners = getCorners(photo_file)#взяли углы с фотки
    cloud = rotateCloud(cloud, alfa, betta, gamma)#повернули облако на тот угол, с которого делалась фотография
    pr=[]
    
    for i in range(len(cloud)):#берём проекцию
        p = Point2(cloud[i].x+270,cloud[i].z+300)#числа взяты так, чтобы проекция рисовалась примерно по центру
        pr.append(p)
   
    conf_pr = getConformity(pr, corners)#сопоставляем проекцию и углы
    triangles = []
    meshes = getMeshes('cube.jpg', corners, 10, 10)#запоминаем нужные грани
    for i in range(len(meshes)):#сопоставляем треугольники с проекции облака и треугольники с фотографии, 
                                #чтобы удобнее было копировать кусочки изображения
        trngl=[]
        
        for j in range(3):
            for k in range(len(corners)):
                if meshes[i][j].x == corners[k].x and meshes[i][j].y == corners[k].y:
                    trngl.append(conf_pr[k])
        triangles.append(trngl)
     
    #наложение текстур
    image = cv2.imread(photo_file) 
    rows,cols,ch = image.shape
    new_image = numpy.zeros(image.shape, numpy.uint8)
    new_image = cv2.bitwise_not(new_image) 
    for i in range(len(meshes)):
        #точки треугольника с фотографии
        x1 = meshes[i][0].x
        y1 = meshes[i][0].y
        x2 = meshes[i][1].x
        y2 = meshes[i][1].y
        x3 = meshes[i][2].x
        y3 = meshes[i][2].y
        pts1 = numpy.float32([[x1,y1],[x2,y2],[x3,y3]])
        roi_corners = numpy.array([[(x1,y1), (x2,y2), (x3,y3)]], dtype=numpy.int32)
        mask = numpy.zeros(image.shape, dtype=numpy.uint8)#маска для фотографии
        #точки треугольника проекции облака
        X1 = triangles[i][0].x
        Y1 = triangles[i][0].y
        X2 = triangles[i][1].x
        Y2 = triangles[i][1].y
        X3 = triangles[i][2].x
        Y3 = triangles[i][2].y       
        pts2 = numpy.float32([[X1,Y1],[X2,Y2],[X3,Y3]])
        roi2_corners = numpy.array([[(X1,Y1), (X2,Y2), (X3,Y3)]], dtype=numpy.int32)
        mask2 = numpy.zeros(new_image.shape, dtype=numpy.uint8)#маска для места, куда вставим изображение
        
        cv2.fillPoly(mask, roi_corners, (255,255,255))#создаём маску
        masked_image = cv2.bitwise_and(image, mask)#применяем маску к фотографии
        M = cv2.getAffineTransform(pts1,pts2)#применяем аффинные преобразования
        warp_affin_img = cv2.warpAffine(masked_image,M,(cols,rows))
        
        cv2.fillPoly(mask2, roi2_corners, (255,255,255))#создаём вторую маску
        mask2 = cv2.bitwise_not(mask2)#инвентируем для обратного эффекта (заполнять нужно то, что вне треугольника)
        new_image = cv2.bitwise_and(new_image, mask2)#применяем маску к прекции
        new_image = cv2.bitwise_or(new_image, warp_affin_img)#объединяем изображения
    cv2.imshow('result',new_image)
        

corners = getCorners('cube.jpg')  
print '=================================='   
for i in range(len(corners)):
    print 'x=', corners[i].x, ' y=', corners[i].y
    
print '===================================='

#print meshe



corners = getCorners('cube.jpg')
#meshe = getTriangles(corners)


#img = cv2.imread('cube_edges.png')
#for i in range(len(meshe)):
#    for j in range(3):
 #       if not pointInTriangle(meshes2[i][j], meshes[i]):
  #          print '!!!'
    
#    cv2.line(img, (meshe[i][0].x,meshe[i][0].y),(meshe[i][1].x,meshe[i][1].y),cv.RGB(255, 0, 25))
#    cv2.line(img, (meshe[i][0].x,meshe[i][0].y),(meshe[i][2].x,meshe[i][2].y),cv.RGB(255, 0, 25))
#    cv2.line(img, (meshe[i][1].x,meshe[i][1].y),(meshe[i][2].x,meshe[i][2].y),cv.RGB(255, 0, 25))
#cv2.imshow('meshes', img)
#cv2.waitKey()

alfa = m.pi/7
betta = 0
gamma = m.pi/4

p1 = Point3(100,100,100)
p2 = Point3(100,100,200)
p3 = Point3(100,200,200)
p4 = Point3(100,200,100)
p5 = Point3(200,100,100)
p6 = Point3(200,100,200)
p7 = Point3(200,200,100)
p8 = Point3(200,200,200)
cloud = [p1,p2,p3,p4,p5,p6,p7,p8]

draw('cube.jpg', cloud, alfa, betta, gamma)


#cloud = rotateCloud(cloud, alfa, betta, gamma)

#pr=[]
#img_white = cv2.imread('white.png')
#for i in range(len(cloud)):
 #   p = Point2(cloud[i].x+270,cloud[i].z+300)
    
  #  pr.append(p)
 
#pr = getConformity(pr, corners)
#meshes = getMeshes('cube.jpg', corners, 10, 10)
#meshes2 = meshes[1] 
#print meshes2
#img = cv2.imread('white.png')

 
#for i in range(len(pr)):
 #   x = int(pr[i].x)
  #  y = int(pr[i].y)
   # cv2.circle(img_white,(x,y),1,cv.RGB(155, 0, 25))
    
#pr_trngl = getTriangles(pr)
#cv2.imshow('proection',img_white)

#for i in range(len(meshe)):
#    img = cv2.imread('white.png')
#    cv2.line(img, (int(pr_trngl[i][0].x),int(pr_trngl[i][0].y)),(int(pr_trngl[i][1].x),int(pr_trngl[i][1].y)),cv.RGB(255, 0, 25))
#    cv2.line(img, (int(pr_trngl[i][0].x),int(pr_trngl[i][0].y)),(int(pr_trngl[i][2].x),int(pr_trngl[i][2].y)),cv.RGB(255, 0, 25))
#    cv2.line(img, (int(pr_trngl[i][1].x),int(pr_trngl[i][1].y)),(int(pr_trngl[i][2].x),int(pr_trngl[i][2].y)),cv.RGB(255, 0, 25))
#    cv2.line(img, (meshe[i][0].x,meshe[i][0].y),(meshe[i][1].x,meshe[i][1].y),cv.RGB(255, 0, 25))
#    cv2.line(img, (meshe[i][0].x,meshe[i][0].y),(meshe[i][2].x,meshe[i][2].y),cv.RGB(255, 0, 25))
#    cv2.line(img, (meshe[i][1].x,meshe[i][1].y),(meshe[i][2].x,meshe[i][2].y),cv.RGB(255, 0, 25))
#    cv2.imshow('meshes', img)
#    cv2.waitKey()

#img = cv2.imread('cube_edges.png')
#img_white = cv2.imread('white.png')





cv.WaitKey()
cv.DestroyAllWindows()
