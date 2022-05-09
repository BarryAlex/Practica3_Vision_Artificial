#Edwing Alexis Casillas Valencia.	Registro: 19110113.	Grupo:7E

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as img

R=1
nombre='E'
color=('r','g','b')
color1=('b','g','r')

def histograma1():
    for i,c in enumerate(color):
        hist=cv2.calcHist([imagen1],[i],None,[256],[0,256])
        plt.plot(hist,color=c)
        plt.xlim([0,256])

def histograma1_eq():
    im1_yuv=cv2.cvtColor(im1,cv2.COLOR_BGR2YUV)
    im1_yuv[:,:,0]=cv2.equalizeHist(im1_yuv[:,:,0])
    equ1=cv2.cvtColor(im1_yuv,cv2.COLOR_YUV2BGR)
    for i,c in enumerate(color1):
        hist_eq1=cv2.calcHist([equ1],[i],None,[256],[0,256])
        plt.plot(hist_eq1,color=c)
        plt.xlim([0,256])

def imagen1_eq():
    im1_yuv=cv2.cvtColor(im1,cv2.COLOR_BGR2YUV)
    im1_yuv[:,:,0]=cv2.equalizeHist(im1_yuv[:,:,0])
    im1_eq1=cv2.cvtColor(im1_yuv,cv2.COLOR_YUV2RGB)
    im1_equ1.imshow(im1_eq1)

def histograma2():
    for i,c in enumerate(color):
        hist_imagen2=cv2.calcHist([imagen2],[i],None,[256],[0,256])
        plt.plot(hist_imagen2,color=c)
        plt.xlim([0,256])

def histograma2_eq():
    im2_yuv=cv2.cvtColor(im2,cv2.COLOR_BGR2YUV)
    im2_yuv[:,:,0]=cv2.equalizeHist(im2_yuv[:,:,0])
    equ2=cv2.cvtColor(im2_yuv,cv2.COLOR_YUV2BGR)
    for i,c in enumerate(color1):
        hist_eq2=cv2.calcHist([equ2],[i],None,[256],[0,256])
        plt.plot(hist_eq2,color=c)
        plt.xlim([0,256])

def imagen2_eq():
    im2_yuv=cv2.cvtColor(im2,cv2.COLOR_BGR2YUV)
    im2_yuv[:,:,0]=cv2.equalizeHist(im2_yuv[:,:,0])
    im2_eq2=cv2.cvtColor(im2_yuv,cv2.COLOR_YUV2RGB)
    im2_equ2.imshow(im2_eq2)

def histograma_op(op):
    for i,c in enumerate(color):
        hist_op=cv2.calcHist([op],[i],None,[256],[0,256])
        plt.plot(hist_op,color=c)
        plt.xlim([0,256])

def histograma_op_eq(op):
    op_yuv=cv2.cvtColor(op,cv2.COLOR_RGB2YUV)
    op_yuv[:,:,0]=cv2.equalizeHist(op_yuv[:,:,0])
    op_eq=cv2.cvtColor(op_yuv,cv2.COLOR_YUV2BGR)
    for i,c in enumerate(color1):
        hist_op_eq=cv2.calcHist([op_eq],[i],None,[256],[0,256])
        plt.plot(hist_op_eq,color=c)
        plt.xlim([0,256])

def imagen_op_eq(op):
    op_yuv=cv2.cvtColor(op,cv2.COLOR_RGB2YUV)
    op_yuv[:,:,0]=cv2.equalizeHist(op_yuv[:,:,0])
    op_equ=cv2.cvtColor(op_yuv,cv2.COLOR_YUV2RGB)
    op_eq.imshow(op_equ)

while(R==1):
    ini=str(input('Presiona la primer letra de tu nombre para comenzar \n'))
    iniup=ini.upper()
    if(iniup==nombre):
        im1=cv2.imread('AoT.png')
        im2=cv2.imread('R&M.png')
        imagen1=cv2.cvtColor(im1,cv2.COLOR_BGR2RGB)
        imagen2=cv2.cvtColor(im2,cv2.COLOR_BGR2RGB)
########################################################################### suma
        suma1=cv2.add(imagen1,imagen2)

        fig1=plt.figure(figsize=(13,3))
        s1=fig1.add_subplot(4,3,1)
        s1.set_title('Imagen 1')
        s1.imshow(imagen1)
        #-------------------------------- histograma
        h1=fig1.add_subplot(4,3,4)
        histograma1()
        #------------------------------------------- ecualizado
        eq1=fig1.add_subplot(4,3,7)
        histograma1_eq()
        #--------------------------------------------- imagen ecualizada
        im1_equ1=fig1.add_subplot(4,3,10)
        imagen1_eq()
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        s2=fig1.add_subplot(4,3,2)
        s2.set_title('Suma add()')
        s2.imshow(suma1)
        #-------------------------------- histograma
        h2=fig1.add_subplot(4,3,5)
        histograma_op(suma1)
        #------------------------------------------- ecualizado
        suma1_equa=fig1.add_subplot(4,3,8)
        histograma_op_eq(suma1)
        #--------------------------------------------- imagen ecualizada
        op_eq=fig1.add_subplot(4,3,11)
        imagen_op_eq(suma1)
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        s3=fig1.add_subplot(4,3,3)
        s3.set_title('Imagen 2')
        s3.imshow(imagen2)
        #-------------------------------- histograma
        h3=fig1.add_subplot(4,3,6)
        histograma2()
        #------------------------------------------- ecualizado
        eq2=fig1.add_subplot(4,3,9)
        histograma2_eq()
        #--------------------------------------------- imagen ecualizada
        im2_equ2=fig1.add_subplot(4,3,12)
        imagen2_eq()
        plt.show()

        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        suma2=imagen1 + imagen2
        fig2=plt.figure(figsize=(13,3))
        s1_1=fig2.add_subplot(4,3,1)
        s1_1.set_title('Imagen 1')
        s1_1.imshow(imagen1)
        #-------------------------------- histograma
        h1_s2=fig2.add_subplot(4,3,4)
        histograma1()
        #------------------------------------------- ecualizado
        eq_s2=fig2.add_subplot(4,3,7)
        histograma1_eq()
        #--------------------------------------------- imagen ecualizada
        im1_equ1=fig2.add_subplot(4,3,10)
        imagen1_eq()
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        s2_1=fig2.add_subplot(4,3,2)
        s2_1.set_title('Suma manual')
        s2_1.imshow(suma2)
        #-------------------------------- histograma
        h2_s2=fig2.add_subplot(4,3,5)
        histograma_op(suma2)
        #------------------------------------------- ecualizado
        suma2_equa=fig2.add_subplot(4,3,8)
        histograma_op_eq(suma2)
        #--------------------------------------------- imagen ecualizada
        op_eq=fig2.add_subplot(4,3,11)
        imagen_op_eq(suma2)
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        s3_1=fig2.add_subplot(4,3,3)
        s3_1.set_title('Imagen 2')
        s3_1.imshow(imagen2)
        #-------------------------------- histograma
        h3_s2=fig2.add_subplot(4,3,6)
        histograma2()
        #------------------------------------------- ecualizado
        eq2_s2=fig2.add_subplot(4,3,9)
        histograma2_eq()
        #--------------------------------------------- imagen ecualizada
        im2_equ2=fig2.add_subplot(4,3,12)
        imagen2_eq()
        plt.show()

        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        suma3=cv2.addWeighted(imagen1,0.3,imagen2,0.7,0)
        fig3=plt.figure(figsize=(13,3))
        s1_2=fig3.add_subplot(4,3,1)
        s1_2.set_title('Imagen 1')
        s1_2.imshow(imagen1)
        #-------------------------------- histograma
        h1_s3=fig3.add_subplot(4,3,4)
        histograma1()
        #------------------------------------------- ecualizado
        eq_s3=fig3.add_subplot(4,3,7)
        histograma1_eq()
        #--------------------------------------------- imagen ecualizada
        im1_equ1=fig3.add_subplot(4,3,10)
        imagen1_eq()
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        s2_2=fig3.add_subplot(4,3,2)
        s2_2.set_title('Suma addWeighted()')
        s2_2.imshow(suma3)
        #-------------------------------- histograma
        hist_s3=fig3.add_subplot(4,3,5)
        histograma_op(suma3)
        #-------------------------------- ecualizado
        hist3_s3=fig3.add_subplot(4,3,8)
        histograma_op_eq(suma3)
        #--------------------------------- imagen ecualizada
        op_eq=fig3.add_subplot(4,3,11)
        imagen_op_eq(suma3)
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        s3_2=fig3.add_subplot(4,3,3)
        s3_2.set_title('Imagen 2')
        s3_2.imshow(imagen2)
        #-------------------------------- histograma
        h3_s3=fig3.add_subplot(4,3,6)
        histograma2()
        #------------------------------------------- ecualizado
        eq2_s3=fig3.add_subplot(4,3,9)
        histograma2_eq()
        #--------------------------------------------- imagen ecualizada
        im2_equ2=fig3.add_subplot(4,3,12)
        imagen2_eq()
        plt.show()


########################################################################## resta
        resta1=cv2.subtract(imagen1,imagen2)
        fig4=plt.figure(figsize=(13,3))
        r1_1=fig4.add_subplot(4,3,1)
        r1_1.set_title('Imagen 1')
        r1_1.imshow(imagen1)
        #----------------------------------- histograma
        r1_1_h=fig4.add_subplot(4,3,4)
        histograma1()
        #----------------------------------- ecualizado
        r1_1_he=fig4.add_subplot(4,3,7)
        histograma1_eq()
        #----------------------------------- imagen ecualizada
        im1_equ1=fig4.add_subplot(4,3,10)
        imagen1_eq()
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        r2_1=fig4.add_subplot(4,3,2)
        r2_1.set_title('Resta subtract()')
        r2_1.imshow(resta1)
        #----------------------------------- histograma
        r2_1_h=fig4.add_subplot(4,3,5)
        histograma_op(resta1)
        #----------------------------------- ecualizado
        r2_1_he=fig4.add_subplot(4,3,8)
        histograma_op_eq(resta1)
        #----------------------------------- imagen ecualizada
        op_eq=fig4.add_subplot(4,3,11)
        imagen_op_eq(resta1)
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        r3_1=fig4.add_subplot(4,3,3)
        r3_1.set_title('Imagen 2')
        r3_1.imshow(imagen2)
        #----------------------------------- histograma
        r3_1_h=fig4.add_subplot(4,3,6)
        histograma2()
        #----------------------------------- ecualizado
        r3_1_he=fig4.add_subplot(4,3,9)
        histograma2_eq()
        #----------------------------------- imagen ecualizada
        im2_equ2=fig4.add_subplot(4,3,12)
        imagen2_eq()
        plt.show()

        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        resta2=cv2.absdiff(imagen1,imagen2)
        fig5=plt.figure(figsize=(13,3))
        r1_2=fig5.add_subplot(4,3,1)
        r1_2.set_title('Imagen 1')
        r1_2.imshow(imagen1)
        #----------------------------------- histograma
        r1_2_h=fig5.add_subplot(4,3,4)
        histograma1()
        #----------------------------------- ecualizado
        r1_2_he=fig5.add_subplot(4,3,7)
        histograma1_eq()
        #----------------------------------- imagen ecualizada
        im1_equ1=fig5.add_subplot(4,3,10)
        imagen1_eq()
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        r2_2=fig5.add_subplot(4,3,2)
        r2_2.set_title('Resta absdiff()')
        r2_2.imshow(resta2)
        #----------------------------------- histograma
        r2_2_h=fig5.add_subplot(4,3,5)
        histograma_op(resta2)
        #----------------------------------- ecualizado
        r2_2_he=fig5.add_subplot(4,3,8)
        histograma_op_eq(resta2)
        #----------------------------------- imagen ecualizada
        op_eq=fig5.add_subplot(4,3,11)
        imagen_op_eq(resta2)
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        r3_2=fig5.add_subplot(4,3,3)
        r3_2.set_title('Imagen 2')
        r3_2.imshow(imagen2)
        #----------------------------------- histograma
        r2_2_h=fig5.add_subplot(4,3,6)
        histograma2()
        #----------------------------------- ecualizado
        r2_2_he=fig5.add_subplot(4,3,9)
        histograma2_eq()
        #----------------------------------- imagen ecualizada
        im2_equ2=fig5.add_subplot(4,3,12)
        imagen2_eq()
        plt.show()

        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        resta3=imagen1-imagen2
        fig6=plt.figure(figsize=(13,3))
        r1_3=fig6.add_subplot(4,3,1)
        r1_3.set_title('Imagen 1')
        r1_3.imshow(imagen1)
        #----------------------------------- histograma
        r1_3_h=fig6.add_subplot(4,3,4)
        histograma1()
        #----------------------------------- ecualizado
        r1_3_he=fig6.add_subplot(4,3,7)
        histograma1_eq()
        #----------------------------------- imagen ecualizada
        im1_equ1=fig6.add_subplot(4,3,10)
        imagen1_eq()
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        r2_3=fig6.add_subplot(4,3,2)
        r2_3.set_title('Resta manual')
        r2_3.imshow(resta3)
        #----------------------------------- histograma
        r2_3_h=fig6.add_subplot(4,3,5)
        histograma_op(resta3)
        #----------------------------------- ecualizado
        r2_3_he=fig6.add_subplot(4,3,8)
        histograma_op_eq(resta3)
        #----------------------------------- imagen ecualizada
        op_eq=fig6.add_subplot(4,3,11)
        imagen_op_eq(resta3)
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        r3_3=fig6.add_subplot(4,3,3)
        r3_3.set_title('Imagen 2')
        r3_3.imshow(imagen2)
        #----------------------------------- histograma
        r3_3_h=fig6.add_subplot(4,3,6)
        histograma2()
        #----------------------------------- ecualizado
        r3_3_he=fig6.add_subplot(4,3,9)
        histograma2_eq()
        #----------------------------------- imagen ecualizada
        im2_equ2=fig6.add_subplot(4,3,12)
        imagen2_eq()
        plt.show()

################################################################# multiplicación
        mul1=cv2.multiply(imagen1,imagen2)
        fig7=plt.figure(figsize=(13,3))
        m1_1=fig7.add_subplot(4,3,1)
        m1_1.set_title('Imagen 1')
        m1_1.imshow(imagen1)
        #----------------------------------- histograma
        m1_1_h=fig7.add_subplot(4,3,4)
        histograma1()
        #----------------------------------- ecualizado
        m1_1_he=fig7.add_subplot(4,3,7)
        histograma1_eq()
        #----------------------------------- imagen ecualizada
        im1_equ1=fig7.add_subplot(4,3,10)
        imagen1_eq()
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        m2_1=fig7.add_subplot(4,3,2)
        m2_1.set_title('Multiplicacion multiply()')
        m2_1.imshow(mul1)
        #----------------------------------- histograma
        m2_1_h=fig7.add_subplot(4,3,5)
        histograma_op(mul1)
        #----------------------------------- ecualizado
        m2_1_he=fig7.add_subplot(4,3,8)
        histograma_op_eq(mul1)
        #----------------------------------- imagen ecualizada
        op_eq=fig7.add_subplot(4,3,11)
        imagen_op_eq(mul1)
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        m3_1=fig7.add_subplot(4,3,3)
        m3_1.set_title('Imagen 2')
        m3_1.imshow(imagen2)
        #----------------------------------- histograma
        m3_1_h=fig7.add_subplot(4,3,6)
        histograma2()
        #----------------------------------- ecualizado
        m3_1_he=fig7.add_subplot(4,3,9)
        histograma2_eq()
        #----------------------------------- imagen ecualizada
        im2_equ2=fig7.add_subplot(4,3,12)
        imagen2_eq()
        plt.show()

        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        mul2=imagen1*imagen2
        fig8=plt.figure(figsize=(13,3))
        m1_2=fig8.add_subplot(4,3,1)
        m1_2.set_title('Imagen 1')
        m1_2.imshow(imagen1)
        #----------------------------------- histograma
        m1_2_h=fig8.add_subplot(4,3,4)
        histograma1()
        #----------------------------------- ecualizado
        m1_2_he=fig8.add_subplot(4,3,7)
        histograma1_eq()
        #----------------------------------- imagen ecualizada
        im1_equ1=fig8.add_subplot(4,3,10)
        imagen1_eq()
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        m2_2=fig8.add_subplot(4,3,2)
        m2_2.set_title('Multiplicacion manual')
        m2_2.imshow(mul2)
        #----------------------------------- histograma
        m2_2_h=fig8.add_subplot(4,3,5)
        histograma_op(mul2)
        #----------------------------------- ecualizado
        m2_2_he=fig8.add_subplot(4,3,8)
        histograma_op_eq(mul2)
        #----------------------------------- imagen ecualizada
        op_eq=fig8.add_subplot(4,3,11)
        imagen_op_eq(mul2)
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        m3_2=fig8.add_subplot(4,3,3)
        m3_2.set_title('Imagen 2')
        m3_2.imshow(imagen2)
        #----------------------------------- histograma
        m3_2_h=fig8.add_subplot(4,3,6)
        histograma2()
        #----------------------------------- ecualizado
        m3_2_he=fig8.add_subplot(4,3,9)
        histograma2_eq()
        #----------------------------------- imagen ecualizada
        im2_equ2=fig8.add_subplot(4,3,12)
        imagen2_eq()
        plt.show()

####################################################################### división
        div1=cv2.divide(imagen1,imagen2)
        fig10=plt.figure(figsize=(13,3))
        d1_1=fig10.add_subplot(4,3,1)
        d1_1.set_title('Imagen 1')
        d1_1.imshow(imagen1)
        #----------------------------------- histograma
        d1_1_h=fig10.add_subplot(4,3,4)
        histograma1()
        #----------------------------------- ecualizado
        d1_1_he=fig10.add_subplot(4,3,7)
        histograma1_eq()
        #----------------------------------- imagen ecualizada
        im1_equ1=fig10.add_subplot(4,3,10)
        imagen1_eq()
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        d2_1=fig10.add_subplot(4,3,2)
        d2_1.set_title('Division divide()')
        d2_1.imshow(div1)
        #----------------------------------- histograma
        d2_1_h=fig10.add_subplot(4,3,5)
        histograma_op(div1)
        #----------------------------------- ecualizado
        d2_1_he=fig10.add_subplot(4,3,8)
        histograma_op_eq(div1)
        #----------------------------------- imagen ecualizada
        op_eq=fig10.add_subplot(4,3,11)
        imagen_op_eq(div1)
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        d3_1=fig10.add_subplot(4,3,3)
        d3_1.set_title('Imagen 2')
        d3_1.imshow(imagen2)
        #----------------------------------- histograma
        d3_1_h=fig10.add_subplot(4,3,6)
        histograma2()
        #----------------------------------- ecualizado
        d3_1_he=fig10.add_subplot(4,3,9)
        histograma2_eq()
        #----------------------------------- imagen ecualizada
        im2_equ2=fig10.add_subplot(4,3,12)
        imagen2_eq()
        plt.show()

        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        divi2=cv2.imread('div2.png')
        div2=cv2.cvtColor(divi2,cv2.COLOR_BGR2RGB)
        fig11=plt.figure(figsize=(13,3))
        d1_2=fig11.add_subplot(4,3,1)
        d1_2.set_title('Imagen 1')
        d1_2.imshow(imagen1)
        #----------------------------------- histograma
        d1_2_h=fig11.add_subplot(4,3,4)
        histograma1()
        #----------------------------------- ecualizado
        d1_2_he=fig11.add_subplot(4,3,7)
        histograma1_eq()
        #----------------------------------- imagen ecualizada
        im1_equ1=fig11.add_subplot(4,3,10)
        imagen1_eq()
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        d2_2=fig11.add_subplot(4,3,2)
        d2_2.set_title('Division manual')
        d2_2.imshow(div2)
        #----------------------------------- histograma
        d2_2_h=fig11.add_subplot(4,3,5)
        histograma_op(div2)
        #----------------------------------- ecualizado
        d2_2_he=fig11.add_subplot(4,3,8)
        histograma_op_eq(div2)
        #----------------------------------- imagen ecualizada
        op_eq=fig11.add_subplot(4,3,11)
        imagen_op_eq(div2)
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        d3_2=fig11.add_subplot(4,3,3)
        d3_2.set_title('Imagen 2')
        d3_2.imshow(imagen2)
        #----------------------------------- histograma
        d3_2_h=fig11.add_subplot(4,3,6)
        histograma2()
        #----------------------------------- ecualizado
        d3_2_he=fig11.add_subplot(4,3,9)
        histograma2_eq()
        #----------------------------------- imagen ecualizada
        im2_equ2=fig11.add_subplot(4,3,12)
        imagen2_eq()
        plt.show()

###################################################################### logaritmo
        c=255/np.log(1+np.max(imagen1))
        log1=c*(np.log(imagen1 + 1))
        log1=np.array(log1,dtype=np.uint8)
        c=255/np.log(1+np.max(imagen2))
        log2=c*(np.log(imagen2 + 1))
        log2=np.array(log2,dtype=np.uint8)
        loga1=cv2.addWeighted(log1,0.5,log2,0.5,0)
        fig13=plt.figure(figsize=(13,3))
        l1_1=fig13.add_subplot(4,3,1)
        l1_1.set_title('Imagen 1')
        l1_1.imshow(imagen1)
        #----------------------------------- histograma
        l1_1_h=fig13.add_subplot(4,3,4)
        histograma1()
        #----------------------------------- ecualizado
        l1_1_he=fig13.add_subplot(4,3,7)
        histograma1_eq()
        #----------------------------------- imagen ecualizada
        im1_equ1=fig13.add_subplot(4,3,10)
        imagen1_eq()
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        l2_1=fig13.add_subplot(4,3,2)
        l2_1.set_title('Logaritmo')
        l2_1.imshow(loga1)
        #----------------------------------- histograma
        l2_1_h=fig13.add_subplot(4,3,5)
        histograma_op(loga1)
        #----------------------------------- ecualizado
        l2_1_he=fig13.add_subplot(4,3,8)
        histograma_op_eq(loga1)
        #----------------------------------- imagen ecualizada
        op_eq=fig13.add_subplot(4,3,11)
        imagen_op_eq(loga1)
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        l3_1=fig13.add_subplot(4,3,3)
        l3_1.set_title('Imagen 2')
        l3_1.imshow(imagen2)
        #----------------------------------- histograma
        l3_1_h=fig13.add_subplot(4,3,6)
        histograma2()
        #----------------------------------- ecualizado
        l3_1_he=fig13.add_subplot(4,3,9)
        histograma2_eq()
        #----------------------------------- imagen ecualizada
        im2_equ2=fig13.add_subplot(4,3,12)
        imagen2_eq()
        plt.show()

####################################################################### potencia
        pot1=cv2.pow(imagen1,2)
        fig16=plt.figure(figsize=(13,3))
        p1_1=fig16.add_subplot(4,3,1)
        p1_1.set_title('Imagen 1')
        p1_1.imshow(imagen1)
        #----------------------------------- histograma
        p1_1_h=fig16.add_subplot(4,3,4)
        histograma1()
        #----------------------------------- ecualizado
        p1_1_he=fig16.add_subplot(4,3,7)
        histograma1_eq()
        #----------------------------------- imagen ecualizada
        im1_equ1=fig16.add_subplot(4,3,10)
        imagen1_eq()
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        p2_1=fig16.add_subplot(4,3,2)
        p2_1.set_title('Potencia pow()')
        p2_1.imshow(pot1)
        #----------------------------------- histograma
        p2_1_h=fig16.add_subplot(4,3,5)
        histograma_op(pot1)
        #----------------------------------- ecualizado
        p2_1_he=fig16.add_subplot(4,3,8)
        histograma_op_eq(pot1)
        #----------------------------------- imagen ecualizada
        op_eq=fig16.add_subplot(4,3,11)
        imagen_op_eq(pot1)
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        p3_1=fig16.add_subplot(4,3,3)
        p3_1.set_title('Imagen 2')
        p3_1.imshow(imagen2)
        #----------------------------------- histograma
        p3_1_h=fig16.add_subplot(4,3,6)
        histograma2()
        #----------------------------------- ecualizado
        p3_1_he=fig16.add_subplot(4,3,9)
        histograma2_eq()
        #----------------------------------- imagen ecualizada
        im2_equ2=fig16.add_subplot(4,3,12)
        imagen2_eq()
        plt.show()

################################################################## raiz cuadrada
        raiz1=cv2.imread('raiz1.png')
        sqrt1=cv2.cvtColor(raiz1,cv2.COLOR_BGR2RGB)
        raiz2=cv2.imread('raiz2.png')
        sqrt2=cv2.cvtColor(raiz2,cv2.COLOR_BGR2RGB)
        fig19=plt.figure(figsize=(13,3))
        root1_1=fig19.add_subplot(4,4,1)
        root1_1.set_title('Imagen 1')
        root1_1.imshow(imagen1)
        #----------------------------------- histograma
        root1_1_h=fig19.add_subplot(4,4,5)
        histograma1()
        #----------------------------------- ecualizado
        root1_1_he=fig19.add_subplot(4,4,9)
        histograma1_eq()
        #----------------------------------- imagen ecualizada
        im1_equ1=fig19.add_subplot(4,4,13)
        imagen1_eq()
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        root2_1=fig19.add_subplot(4,4,2)
        root2_1.set_title('Raiz manual 1')
        root2_1.imshow(sqrt1)
        #----------------------------------- histograma
        root2_1_h=fig19.add_subplot(4,4,6)
        histograma_op(sqrt1)
        #----------------------------------- ecualizado
        root2_1_he=fig19.add_subplot(4,4,10)
        histograma_op_eq(sqrt1)
        #----------------------------------- imagen ecualizada
        op_eq=fig19.add_subplot(4,4,14)
        imagen_op_eq(sqrt1)
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        root3_1=fig19.add_subplot(4,4,3)
        root3_1.set_title('Raiz manual 2')
        root3_1.imshow(sqrt2)
        #----------------------------------- histograma
        root3_1_h=fig19.add_subplot(4,4,7)
        histograma_op(sqrt2)
        #----------------------------------- ecualizado
        root3_1_he=fig19.add_subplot(4,4,11)
        histograma_op_eq(sqrt2)
        #----------------------------------- imagen ecualizada
        op_eq=fig19.add_subplot(4,4,15)
        imagen_op_eq(sqrt2)
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        root4_1=fig19.add_subplot(4,4,4)
        root4_1.set_title('Imagen 2')
        root4_1.imshow(imagen2)
        #----------------------------------- histograma
        root4_1_h=fig19.add_subplot(4,4,8)
        histograma2()
        #----------------------------------- ecualizado
        root4_1_he=fig19.add_subplot(4,4,12)
        histograma2_eq()
        #----------------------------------- imagen ecualizada
        im2_equ2=fig19.add_subplot(4,4,16)
        imagen2_eq()
        plt.show()

        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        raiz3=cv2.imread('raiz3.png')
        sqrt3=cv2.cvtColor(raiz3,cv2.COLOR_BGR2RGB)
        raiz4=cv2.imread('raiz4.png')
        sqrt4=cv2.cvtColor(raiz4,cv2.COLOR_BGR2RGB)
        fig20=plt.figure(figsize=(13,3))
        root1_2=fig20.add_subplot(4,4,1)
        root1_2.set_title('Imagen 1')
        root1_2.imshow(imagen1)
        #----------------------------------- histograma
        root1_2_h=fig20.add_subplot(4,4,5)
        histograma1()
        #----------------------------------- ecualizado
        root1_2_he=fig20.add_subplot(4,4,9)
        histograma1_eq()
        #----------------------------------- imagen ecualizada
        im1_equ1=fig20.add_subplot(4,4,13)
        imagen1_eq()
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        root2_2=fig20.add_subplot(4,4,2)
        root2_2.set_title('Raiz np.sqrt() 1')
        root2_2.imshow(sqrt3)
        #----------------------------------- histograma
        root2_2_h=fig20.add_subplot(4,4,6)
        histograma_op(sqrt3)
        #----------------------------------- ecualizado
        root2_2_he=fig20.add_subplot(4,4,10)
        histograma_op_eq(sqrt3)
        #----------------------------------- imagen ecualizada
        op_eq=fig20.add_subplot(4,4,14)
        imagen_op_eq(sqrt3)
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        root2_2=fig20.add_subplot(4,4,3)
        root2_2.set_title('Raiz np.sqrt() 2')
        root2_2.imshow(sqrt4)
        #----------------------------------- histograma
        root2_2_h=fig20.add_subplot(4,4,7)
        histograma_op(sqrt4)
        #----------------------------------- ecualizado
        root2_2_he=fig20.add_subplot(4,4,11)
        histograma_op_eq(sqrt4)
        #----------------------------------- imagen ecualizada
        op_eq=fig20.add_subplot(4,4,15)
        imagen_op_eq(sqrt4)
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        root4_2=fig20.add_subplot(4,4,4)
        root4_2.set_title('Imagen 2')
        root4_2.imshow(imagen2)
        #----------------------------------- histograma
        root4_2_h=fig20.add_subplot(4,4,8)
        histograma2()
        #----------------------------------- ecualizado
        root4_2_he=fig20.add_subplot(4,4,12)
        histograma2_eq()
        #----------------------------------- imagen ecualizada
        im2_equ2=fig20.add_subplot(4,4,16)
        imagen2_eq()
        plt.show()

####################################################################### derivada
        de1=cv2.imread('Derivada1.png')
        de2=cv2.imread('Derivada2.png')
        der1=cv2.cvtColor(de1,cv2.COLOR_BGR2RGB)
        der2=cv2.cvtColor(de2,cv2.COLOR_BGR2RGB)
        fig22=plt.figure(figsize=(13,3))
        deri1=fig22.add_subplot(4,4,1)
        deri1.set_title('Imagen 2')
        deri1.imshow(imagen1)
        #----------------------------------- histograma
        deri1_h=fig22.add_subplot(4,4,5)
        histograma1()
        #----------------------------------- ecualizado
        deri1_he=fig22.add_subplot(4,4,9)
        histograma1_eq()
        #----------------------------------- imagen ecualizada
        im1_equ1=fig22.add_subplot(4,4,13)
        imagen1_eq()
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        deri2=fig22.add_subplot(4,4,2)
        deri2.set_title('Derivada Sobel 1')
        deri2.imshow(der1)
        #----------------------------------- histograma
        deri2_h=fig22.add_subplot(4,4,6)
        histograma_op(der1)
        #----------------------------------- ecualizado
        deri2_he=fig22.add_subplot(4,4,10)
        histograma_op_eq(der1)
        #----------------------------------- imagen ecualizada
        op_eq=fig22.add_subplot(4,4,14)
        imagen_op_eq(der1)
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        deri3=fig22.add_subplot(4,4,3)
        deri3.set_title('Derivada Sobel 2')
        deri3.imshow(der2)
        #----------------------------------- histograma
        deri3_h=fig22.add_subplot(4,4,7)
        histograma_op(der2)
        #----------------------------------- ecualizado
        deri3_he=fig22.add_subplot(4,4,11)
        histograma_op_eq(der2)
        #----------------------------------- imagen ecualizada
        op_eq=fig22.add_subplot(4,4,15)
        imagen_op_eq(der2)
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        deri4=fig22.add_subplot(4,4,4)
        deri4.set_title('Imagen 2')
        deri4.imshow(imagen2)
        #----------------------------------- histograma
        deri4_h=fig22.add_subplot(4,4,8)
        histograma2()
        #----------------------------------- ecualizado
        deri4_he=fig22.add_subplot(4,4,12)
        histograma2_eq()
        #----------------------------------- imagen ecualizada
        im2_equ2=fig22.add_subplot(4,4,16)
        imagen2_eq()
        plt.show()
##################################################################### conjunción
        conj1=cv2.bitwise_and(imagen1,imagen2)
        fig25=plt.figure(figsize=(13,3))
        conj1_1=fig25.add_subplot(4,3,1)
        conj1_1.set_title('Imagen 1')
        conj1_1.imshow(imagen1)
        #----------------------------------- histograma
        conj1_1_h=fig25.add_subplot(4,3,4)
        histograma1()
        #----------------------------------- ecualizado
        conj1_1_he=fig25.add_subplot(4,3,7)
        histograma1_eq()
        #----------------------------------- imagen ecualizada
        im1_equ1=fig25.add_subplot(4,3,10)
        imagen1_eq()
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        conj2_1=fig25.add_subplot(4,3,2)
        conj2_1.set_title('Conjuncion bitwise_and()')
        conj2_1.imshow(conj1)
        #----------------------------------- histograma
        conj2_1_h=fig25.add_subplot(4,3,5)
        histograma_op(conj1)
        #----------------------------------- ecualizado
        conj2_1_he=fig25.add_subplot(4,3,8)
        histograma_op_eq(conj1)
        #----------------------------------- imagen ecualizada
        op_eq=fig25.add_subplot(4,3,11)
        imagen_op_eq(conj1)
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        conj3_1=fig25.add_subplot(4,3,3)
        conj3_1.set_title('Imagen 2')
        conj3_1.imshow(imagen2)
        #----------------------------------- histograma
        conj3_1_h=fig25.add_subplot(4,3,6)
        histograma2()
        #----------------------------------- ecualizado
        conj3_1_he=fig25.add_subplot(4,3,9)
        histograma2_eq()
        #----------------------------------- imagen ecualizada
        im2_equ2=fig25.add_subplot(4,3,12)
        imagen2_eq()
        plt.show()
##################################################################### disyunción
        disy1=cv2.bitwise_or(imagen1,imagen2)
        fig28=plt.figure(figsize=(13,3))
        disy1_1=fig28.add_subplot(4,3,1)
        disy1_1.set_title('Imagen 1')
        disy1_1.imshow(imagen1)
        #----------------------------------- histograma
        disy1_1_h=fig28.add_subplot(4,3,4)
        histograma1()
        #----------------------------------- ecualizado
        disy1_1_he=fig28.add_subplot(4,3,7)
        histograma1_eq()
        #----------------------------------- imagen ecualizada
        im1_equ1=fig28.add_subplot(4,3,10)
        imagen1_eq()
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        disy2_1=fig28.add_subplot(4,3,2)
        disy2_1.set_title('Disyuncion bitwise_or()')
        disy2_1.imshow(disy1)
        #----------------------------------- histograma
        disy2_1_h=fig28.add_subplot(4,3,5)
        histograma_op(disy1)
        #----------------------------------- ecualizado
        disy2_1_he=fig28.add_subplot(4,3,8)
        histograma_op_eq(disy1)
        #----------------------------------- imagen ecualizada
        op_eq=fig28.add_subplot(4,3,11)
        imagen_op_eq(disy1)
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        disy3_1=fig28.add_subplot(4,3,3)
        disy3_1.set_title('Imagen 2')
        disy3_1.imshow(imagen2)
        #----------------------------------- histograma
        disy3_1_h=fig28.add_subplot(4,3,6)
        histograma2()
        #----------------------------------- ecualizado
        disy3_1_he=fig28.add_subplot(4,3,9)
        histograma2_eq()
        #----------------------------------- imagen ecualizada
        im2_equ2=fig28.add_subplot(4,3,12)
        imagen2_eq()
        plt.show()
####################################################################### negación
        not1=cv2.bitwise_not(imagen1)
        fig31=plt.figure(figsize=(13,3))
        not1_1=fig31.add_subplot(4,3,1)
        not1_1.set_title('Imagen 1')
        not1_1.imshow(imagen1)
        #----------------------------------- histograma
        not1_1_h=fig31.add_subplot(4,3,4)
        histograma1()
        #----------------------------------- ecualizado
        not1_1_he=fig31.add_subplot(4,3,7)
        histograma1_eq()
        #----------------------------------- imagen ecualizada
        im1_equ1=fig31.add_subplot(4,3,10)
        imagen1_eq()
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        not2_1=fig31.add_subplot(4,3,2)
        not2_1.set_title('Negacion bitwise_not()')
        not2_1.imshow(not1)
        #----------------------------------- histograma
        not2_1_h=fig31.add_subplot(4,3,5)
        histograma_op(not1)
        #----------------------------------- ecualizado
        not2_1_he=fig31.add_subplot(4,3,8)
        histograma_op_eq(not1)
        #----------------------------------- imagen ecualizada
        op_eq=fig31.add_subplot(4,3,11)
        imagen_op_eq(not1)
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        not3_1=fig31.add_subplot(4,3,3)
        not3_1.set_title('Imagen 2')
        not3_1.imshow(imagen2)
        #----------------------------------- histograma
        not3_1_h=fig31.add_subplot(4,3,6)
        histograma2()
        #----------------------------------- ecualizado
        not3_1_he=fig31.add_subplot(4,3,9)
        histograma2_eq()
        #----------------------------------- imagen ecualizada
        im2_equ2=fig31.add_subplot(4,3,12)
        imagen2_eq()
        plt.show()

##################################################################### traslación
        ancho=imagen1.shape[1]
        alto=imagen1.shape[0]
        M2=np.float32([[1,0,100],[0,1,150]])
        tra1=cv2.warpAffine(imagen1,M2,(ancho,alto))
        fig34=plt.figure(figsize=(13,3))
        tra1_1=fig34.add_subplot(4,3,1)
        tra1_1.set_title('Imagen 1')
        tra1_1.imshow(imagen1)
        #----------------------------------- histograma
        tra1_1_h=fig34.add_subplot(4,3,4)
        histograma1()
        #----------------------------------- ecualizado
        tra1_1_he=fig34.add_subplot(4,3,7)
        histograma1_eq()
        #----------------------------------- imagen ecualizada
        im1_equ1=fig34.add_subplot(4,3,10)
        imagen1_eq()
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        tra2_1=fig34.add_subplot(4,3,2)
        tra2_1.set_title('Traslacion')
        tra2_1.imshow(tra1)
        #----------------------------------- histograma
        tra2_1_h=fig34.add_subplot(4,3,5)
        histograma_op(tra1)
        #----------------------------------- ecualizado
        tra2_1_he=fig34.add_subplot(4,3,8)
        histograma_op_eq(tra1)
        #----------------------------------- imagen ecualizada
        op_eq=fig34.add_subplot(4,3,11)
        imagen_op_eq(tra1)
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        tra3_1=fig34.add_subplot(4,3,3)
        tra3_1.set_title('Imagen 2')
        tra3_1.imshow(imagen2)
        #----------------------------------- histograma
        tra3_1_h=fig34.add_subplot(4,3,6)
        histograma2()
        #----------------------------------- ecualizado
        tra3_1_he=fig34.add_subplot(4,3,9)
        histograma2_eq()
        #----------------------------------- imagen ecualizada
        im2_equ2=fig34.add_subplot(4,3,12)
        imagen2_eq()
        plt.show()

####################################################################### escalado
        scale_percent=50        #Porcentaje en el que se redimencionala imagen
		#calculo de dimensiones deseada
        width=int(imagen1.shape[1]*scale_percent/100)
        height=int(imagen1.shape[0]*scale_percent/100)
		#tamaño
        dsize=(width,height)
		#cambiar el tamaño de la imagen
        output=cv2.resize(imagen1,dsize)
        cv2.imwrite("Escalado_Shingeki_R&M.png",output)
        fig37=plt.figure(figsize=(13,3))
        esc1_1=fig37.add_subplot(4,3,1)
        esc1_1.set_title('Imagen 1')
        esc1_1.imshow(imagen1)
        #----------------------------------- histograma
        esc1_1_h=fig37.add_subplot(4,3,4)
        histograma1()
        #----------------------------------- ecualizado
        esc1_1_he=fig37.add_subplot(4,3,7)
        histograma1_eq()
        #----------------------------------- imagen ecualizada
        im1_equ1=fig37.add_subplot(4,3,10)
        imagen1_eq()
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        esc2_1=fig37.add_subplot(4,3,2)
        esc2_1.set_title('Escalado')
        esc2_1.imshow(output)
        #----------------------------------- histograma
        esc2_1_h=fig37.add_subplot(4,3,5)
        histograma_op(output)
        #----------------------------------- ecualizado
        esc2_1_he=fig37.add_subplot(4,3,8)
        histograma_op_eq(output)
        #----------------------------------- imagen ecualizada
        op_eq=fig37.add_subplot(4,3,11)
        imagen_op_eq(output)
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        esc3_1=fig37.add_subplot(4,3,3)
        esc3_1.set_title('Imagen 2')
        esc3_1.imshow(imagen2)
        #----------------------------------- histograma
        esc3_1_h=fig37.add_subplot(4,3,6)
        histograma2()
        #----------------------------------- ecualizado
        esc3_1_he=fig37.add_subplot(4,3,9)
        histograma2_eq()
        #----------------------------------- imagen ecualizada
        im2_equ2=fig37.add_subplot(4,3,12)
        imagen2_eq()
        plt.show()

####################################################################### rotación
        rot1=cv2.getRotationMatrix2D((ancho//2,alto//2),15,1)
        rot=cv2.warpAffine(imagen1,rot1,(ancho,alto))
        fig40=plt.figure(figsize=(13,3))
        rot1_1=fig40.add_subplot(4,3,1)
        rot1_1.set_title('Imagen 1')
        rot1_1.imshow(imagen1)
        #----------------------------------- histograma
        rot1_1_h=fig40.add_subplot(4,3,4)
        histograma1()
        #----------------------------------- ecualizado
        rot1_1_he=fig40.add_subplot(4,3,7)
        histograma1_eq()
        #----------------------------------- imagen ecualizada
        im1_equ1=fig40.add_subplot(4,3,10)
        imagen1_eq()
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        rot2_1=fig40.add_subplot(4,3,2)
        rot2_1.set_title('Rotacion')
        rot2_1.imshow(rot)
        #----------------------------------- histograma
        rot2_1_h=fig40.add_subplot(4,3,5)
        histograma_op(rot)
        #----------------------------------- ecualizado
        rot2_1_he=fig40.add_subplot(4,3,8)
        histograma_op_eq(rot)
        #----------------------------------- imagen ecualizada
        op_eq=fig40.add_subplot(4,3,11)
        imagen_op_eq(rot)
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        rot3_1=fig40.add_subplot(4,3,3)
        rot3_1.set_title('Imagen 2')
        rot3_1.imshow(imagen2)
        #----------------------------------- histograma
        rot3_1_h=fig40.add_subplot(4,3,6)
        histograma2()
        #----------------------------------- ecualizado
        rot3_1_he=fig40.add_subplot(4,3,9)
        histograma2_eq()
        #----------------------------------- imagen ecualizada
        im2_equ2=fig40.add_subplot(4,3,12)
        imagen2_eq()
        plt.show()

############################################################### traslación a fin
        pts1=np.float32([[100,400],[400,100],[100,100]])
        pts2=np.float32([[50,300],[400,200],[80,150]])
        M3=cv2.getAffineTransform(pts1,pts2)
        traf=cv2.warpAffine(imagen1,M3,(alto,ancho))
        fig43=plt.figure(figsize=(13,3))
        traf1_1=fig43.add_subplot(4,3,1)
        traf1_1.set_title('Imagen 1')
        traf1_1.imshow(imagen1)
        #----------------------------------- histograma
        traf1_1_h=fig43.add_subplot(4,3,4)
        histograma1()
        #----------------------------------- ecualizado
        traf1_1_he=fig43.add_subplot(4,3,7)
        histograma1_eq()
        #----------------------------------- imagen ecualizada
        im1_equ1=fig43.add_subplot(4,3,10)
        imagen1_eq()
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        traf2_1=fig43.add_subplot(4,3,2)
        traf2_1.set_title('Traslaciona afin')
        traf2_1.imshow(traf)
        #----------------------------------- histograma
        traf2_1_h=fig43.add_subplot(4,3,5)
        histograma_op(traf)
        #----------------------------------- ecualizado
        traf2_1_he=fig43.add_subplot(4,3,8)
        histograma_op_eq(traf)
        #----------------------------------- imagen ecualizada
        op_eq=fig43.add_subplot(4,3,11)
        imagen_op_eq(traf)
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        traf3_1=fig43.add_subplot(4,3,3)
        traf3_1.set_title('Imagen 2')
        traf3_1.imshow(imagen2)
        #----------------------------------- histograma
        traf3_1_h=fig43.add_subplot(4,3,6)
        histograma2()
        #----------------------------------- ecualizado
        traf3_1_he=fig43.add_subplot(4,3,9)
        histograma2_eq()
        #----------------------------------- imagen ecualizada
        im2_equ2=fig43.add_subplot(4,3,12)
        imagen2_eq()
        plt.show()

#################################################################### transpuesta
        trans=cv2.getRotationMatrix2D((ancho/2,alto/2),90,1)
        trans1=cv2.warpAffine(imagen1,trans,(ancho,alto))
        trans2=cv2.warpAffine(imagen2,trans,(ancho,alto))
        fig47=plt.figure(figsize=(13,3))
        transp1_2=fig47.add_subplot(4,4,1)
        transp1_2.set_title('Imagen 1')
        transp1_2.imshow(imagen1)
        #----------------------------------- histograma
        transp1_2_h=fig47.add_subplot(4,4,5)
        histograma1()
        #----------------------------------- ecualizado
        transp1_2_he=fig47.add_subplot(4,4,9)
        histograma1_eq()
        #----------------------------------- imagen ecualizada
        im1_equ1=fig47.add_subplot(4,4,13)
        imagen1_eq()
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        transp2_2=fig47.add_subplot(4,4,2)
        transp2_2.set_title('Transpuesta 1')
        transp2_2.imshow(trans1)
        #----------------------------------- histograma
        transp2_2_h=fig47.add_subplot(4,4,6)
        histograma_op(trans1)
        #----------------------------------- ecualizado
        transp2_2_he=fig47.add_subplot(4,4,10)
        histograma_op_eq(trans1)
        #----------------------------------- imagen ecualizada
        op_eq=fig47.add_subplot(4,4,14)
        imagen_op_eq(trans1)
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        transp3_2=fig47.add_subplot(4,4,3)
        transp3_2.set_title('Transpuesta 2')
        transp3_2.imshow(trans2)
        #----------------------------------- histograma
        transp3_2_h=fig47.add_subplot(4,4,7)
        histograma_op(trans2)
        #----------------------------------- ecualizado
        transp3_2_he=fig47.add_subplot(4,4,11)
        histograma_op_eq(trans2)
        #----------------------------------- imagen ecualizada
        op_eq=fig47.add_subplot(4,4,15)
        imagen_op_eq(trans2)
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        transp4_2=fig47.add_subplot(4,4,4)
        transp4_2.set_title('Imagen 2')
        transp4_2.imshow(imagen2)
        #----------------------------------- histograma
        transp4_2_h=fig47.add_subplot(4,4,8)
        histograma2()
        #----------------------------------- ecualizado
        transp4_2_he=fig47.add_subplot(4,4,12)
        histograma2_eq()
        #----------------------------------- imagen ecualizada
        im2_equ2=fig47.add_subplot(4,4,16)
        imagen2_eq()
        plt.show()
##################################################################### proyección
        pts3=np.float32([[200, 0],[350,0],[28,200],[350,210]])
        pts4=np.float32([[0,0],[480,0],[0,270],[480,270]])
        matriz=cv2.getPerspectiveTransform(pts3,pts4)
        proy=cv2.warpPerspective(imagen1,matriz,(480,270))
        fig49=plt.figure(figsize=(13,3))
        proy1_1=fig49.add_subplot(4,3,1)
        proy1_1.set_title('Imagen 1')
        proy1_1.imshow(imagen1)
        #----------------------------------- histograma
        proy1_1_h=fig49.add_subplot(4,3,4)
        histograma1()
        #----------------------------------- ecualizado
        proy1_1_he=fig49.add_subplot(4,3,7)
        histograma1_eq()
        #----------------------------------- imagen ecualizada
        im1_equ1=fig49.add_subplot(4,3,10)
        imagen1_eq()
        #oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        proy2_1=fig49.add_subplot(4,3,2)
        proy2_1.set_title('Proyeccion')
        proy2_1.imshow(proy)
        #----------------------------------- histograma
        proy2_1_h=fig49.add_subplot(4,3,5)
        histograma_op(proy)
        #----------------------------------- ecualizado
        proy2_1_he=fig49.add_subplot(4,3,8)
        histograma_op_eq(proy)
        #----------------------------------- imagen ecualizada
        op_eq=fig49.add_subplot(4,3,11)
        imagen_op_eq(proy)
        #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
        proy3_1=fig49.add_subplot(4,3,3)
        proy3_1.set_title('Imagen 2')
        proy3_1.imshow(imagen2)
        #----------------------------------- histograma
        proy3_1_h=fig49.add_subplot(4,3,6)
        histograma2()
        #----------------------------------- ecualizado
        proy3_1_he=fig49.add_subplot(4,3,9)
        histograma2_eq()
        #----------------------------------- imagen ecualizada
        im2_equ2=fig49.add_subplot(4,3,12)
        imagen2_eq()
        plt.show()

    else:
        print("")

    R=int(input("Presione cualquier numero diferente de 1 para salir\n"))
    print("")

    if(R==1):
        os.system("cls")
    else:
        break
