# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:13:58 2015

@author: klek
"""


from cv2 import *

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
       
def sortPoints(cloud, x, y, z):
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
       
   

def getCorners(inputFile):
    
    foundPoints =[] 
    
    imcolor = cv.LoadImage(inputFile)
    image = cv.LoadImage(inputFile,cv.CV_LOAD_IMAGE_GRAYSCALE)
    cornerMap = cv.CreateMat(image.height, image.width, cv.CV_32FC1)
    # OpenCV corner detection
    cv.CornerHarris(image,cornerMap,3)
    
    for y in range(0, image.height):
        for x in range(0, image.width):
            harris = cv.Get2D(cornerMap, y, x) # get the x,y value
            # check the corner detector response
            if harris[0] > 10e-06:
                # draw a small circle on the original image
                cv.Circle(imcolor,(x,y),10,cv.RGB(155, 0, 25))
                point = Point2(x,y)
                foundPoints.append(point)
                print('x =', x, ' y =', y) 
                
                cv.NamedWindow('Harris', cv.CV_WINDOW_AUTOSIZE)
                cv.ShowImage('Harris', imcolor) # show the image
                #print foundPoints
    used = []
    for i in range(len(foundPoints)):
        used.append(0)
        result = []
        
    for i in range(len(foundPoints)):
        points = []
            
        for j in range(len(foundPoints)):
            l1 = foundPoints[i].x-foundPoints[j].x
            l2 = foundPoints[i].y-foundPoints[j].y
            if (m.fabs(l1) <= 5) and (m.fabs(l2) <= 5) and (used[j]==0):
                points.append(foundPoints[j])
                used[j]=1
        sx = 0
        sy = 0
        if len(points)!=0:
            for k in range(0,len(points)):
                sx = sx+points[k].x
                sy = sy+points[k].y
            p = Point2(sx/len(points),sy/len(points))
            result.append(p)
    return result
                            
def Proection(cloud,alfa, betta):
    M1 = numpy.array([[1,0,0],
                      [0,m.cos(alfa),m.sin(alfa)],
                       [0, -m.sin(alfa),m.cos(alfa)]])
    M2 = numpy.array([[m.cos(betta),0, -m.sin(betta)],
                       [0,1,0],
                       [m.sin(betta),0,m.cos(betta)]])
     
                   
    I = numpy.array([[1,0,0],
                     [0,1,0],
                     [0,0,0]])
                     
    M = [] 
                     
    for i in range(len(cloud)):
        p = numpy.array([cloud[i].x,cloud[i].y,cloud[i].z])
    
        M.append(numpy.dot(numpy.dot(numpy.dot(M1,M2),p), I))
    
    result = []    
    for i in range(len(M)):
        p2 = Point2(M[i][0],M[i][1])
        result.append(p2)
    return(result)       


result = getCorners('cube.jpg')  
print '=================================='   
for i in range(len(result)):
    print 'x=', result[i].x, ' y=', result[i].y
    

image = cv.LoadImage('cube.jpg')
cv.NamedWindow('Corners', cv.CV_WINDOW_AUTOSIZE)
for i in range(len(result)):
    
    cx = int(result[i].x)
    cy = int(result[i].y)
    cv.Circle(image,(cx,cy),1,cv.RGB(255, 0, 25))
cv.ShowImage('Corners', image)
#cv.WaitKey()
#image = cv.LoadImage("white.png")
#a = image.height -100
#b = image.width -100
#p1 = Point3(100,100,100)
#p2 = Point3(100,100,200)
#p3 = Point3(100,200,200)
#p4 = Point3(100,200,100)
#p5 = Point3(200,100,100)
#p6 = Point3(200,100,200)
#p7 = Point3(200,200,100)
#p8 = Point3(200,200,200)
#cloud = [p1,p2,p3,p4,p5,p6,p7,p8]
#cloud = sortPoints(cloud, 1000, 1000, 1000)
#for i in range(len(cloud)):
 #   print 'x=', cloud[i].x, ' y=', cloud[i].y, ' z=', cloud[i].z 
#al = m.asin(m.tan(m.pi/6))
#bet = m.pi/10
#cloud2 = Proection(cloud, al, bet)
#for i in range(len(cloud2)):
#    print "x =", cloud2[i].x, "y =", cloud2[i].y

#cv.NamedWindow("Cloud", 1)



#for i in range(len(cloud2)):
    
#    cx = int(cloud2[i].x)+100
#    cy = int(cloud2[i].y)
#    cv.Circle(image,(cx,cy),2,cv.RGB(0, 0, 0))

#cv.ShowImage("Cloud", image)
cv.WaitKey()

