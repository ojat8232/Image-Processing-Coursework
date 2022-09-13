import cv2
import numpy as np
from numpy import random
import math
import cmath
from scipy import interpolate



#inputFileName is the file name of the input file
#outputFileName is the name of the file to be saved
#blendCo is the blending coefficent between the mask and the picture
#darkCo is the darkenig coefficent
#mode is set to 1 for simple light leak and 2 for rainbow light leak
#example call: problem1("face1.jpg", "test.jpg", 21, 7, 2)
def problem1(inputFileName, outputFileName, blendCo, darkCo, mode):
    input = cv2.imread(inputFileName, cv2.IMREAD_COLOR)
    if input is not None:

        #simple light leak
        if mode == 1:
            filter = cv2.imread("LightLeakFilter.png", cv2.IMREAD_COLOR)
            filter = cv2.resize(filter, (input.shape[0], input.shape[1]))
            
            for y in range(input.shape[0]):
                for x in range(input.shape[1]):
                    for c in range(input.shape[2]):
                        input[y, x, c ] = input[y, x, c]/ darkCo
                        if filter[y, x, c] != 0:
                            input[y, x, c] = input[y, x, c] + blendCo * filter[y, x, c] /255
                          
        #rainbow light leak
        elif mode ==2:
            filter = cv2.imread("rainbowFilter.png", cv2.IMREAD_COLOR)
            filter = cv2.resize(filter, (input.shape[0], input.shape[1]))
            for y in range(input.shape[0]):
                for x in range(input.shape[1]):
                    for c in range(input.shape[2]):
                        input[y, x, c ] = input[y, x, c]/ darkCo
                        if filter[y, x, c] != 0:
                            input[y, x, c] = input[y, x, c] + blendCo * filter[y, x, c] /255

        cv2.imshow("Image", input)
        cv2.imwrite(outputFileName, input) 
        cv2.waitKey()
    else:
        print("The image failed to load")

#inputFileName is the file name of the input file
#outputFileName is the name of the file to be saved
#blendCo is the blending coefficent between the mask and the picture
#maskSize is the size of the mask for the motion blur
#mode = 1 for greyscale and 2 for colour
def problem2(inputFileName, outputFileName, blendCo, maskSize, mode):
    #blendCo = 0.4
    #mode = 2
    #maskSize = 5
    input = cv2.imread(inputFileName, cv2.IMREAD_GRAYSCALE)
    noise = random.randint(0, 255, input.shape, np.uint8)
    mask = np.zeros((maskSize, maskSize))
    k = int(maskSize / 2)
    mask[k, :] = np.ones(maskSize)
    mask /= maskSize
    noise = cv2.filter2D(noise, -1, mask)

    inverse = cv2.bitwise_not(input)
    noise = cv2.multiply(noise, blendCo)
    if mode == 2:
        inverse = cv2.cvtColor(inverse, cv2.COLOR_GRAY2BGR)
        input = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
        noise= cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)

        b, g, r = cv2.split(inverse)
        r = cv2.multiply(r, 1 - blendCo)
        b = cv2.multiply(b, 1 - blendCo)
        g =  np.full( g.shape, 255, np.uint8)
        b1, g1, r1 = cv2.split(input)
        g1 = np.full( g.shape, 0, np.uint8)
        #temp = cv2.add(noise, inverse)
        temp = cv2.merge((b, g,r))
        temp = cv2.add(noise, temp)
        input = cv2.merge((b1, g1, r1))
    elif mode == 1:
        inverse = cv2.multiply(inverse, 1 - blendCo)
        temp = cv2.add(noise, inverse)

    output = cv2.divide(input, temp, scale=256.0)

    cv2.imshow("Image", output)
    cv2.imwrite(outputFileName, output) 
    cv2.waitKey()     


#inputFileName is the file name of the input file
#outputFileName is the name of the file to be saved 
def problem3(inputFileName, outputFileName):

    def spl(x, y):
        spline = interpolate.UnivariateSpline(x, y)
        return spline( range( 256 ) )

    input = cv2.imread(inputFileName, cv2.IMREAD_COLOR)

    #mask = np.ones((3, 3))
    #input = cv2.filter2D(input, -1, mask)

    b, g, r = cv2.split(input)

    r = cv2.LUT(r, spl([0, 64, 128, 256], [5, 80, 160, 256])).astype(np.uint8)
    g = cv2.LUT(g, spl([0, 64, 128, 256], [0, 60, 120, 256])).astype(np.uint8)
    b = cv2.LUT(b, spl([0, 64, 128, 256], [0, 50, 110, 240])).astype(np.uint8)

    output = cv2.merge((b, g, r))

    cv2.imshow("Image", output)
    cv2.imwrite(outputFileName, output) 
    cv2.waitKey()


#inputFileName is the file name of the input file
#outputFileName is the name of the file to be saved 
#swirlAmount is the number of rotations
#radius is the radius of the swirl area
#mode = 1 is nearest neighbour, mode = 2 is bilineal interpolation
#prefilter = 1 is prefilter
#Example call: problem4("face1.jpg", "test4.jpg", 0.3, 200, 1, 0)
def problem4(inputFileName, outputFileName, swirlAmount, radius, mode, prefilter):

    input = cv2.imread(inputFileName, cv2.IMREAD_COLOR)
    output = input.copy()

    if prefilter == 1:
        print("prefilter")

    for y in range(input.shape[0]):
            for x in range(input.shape[1]):
                    
                    t = complex(x - input.shape[0] / 2, y - input.shape[1] / 2)
                    r, p = cmath.polar(t)
                    if r < radius and r != 0:
                        p = p + (math.pi * swirlAmount * (1 - (r / radius)) * 2)
                        k = cmath.rect(r, p)

                        #nearest neighbour
                        if mode == 1:
                            a = round(k.real + input.shape[0] / 2)
                            b = round(k.imag + input.shape[1] / 2)
                            if a < input.shape[0] and b < input.shape[1]:
                                output[x, y] = input[a, b]

                        #bilineal interpolation
                        if mode == 2:
                            a = k.real + input.shape[0] / 2
                            b = k.imag + input.shape[1] / 2
                            if math.ceil(a) < input.shape[0] and math.ceil(b) < input.shape[1]:
                                c = x- int(x)
                                d = y-int(y)
                                i = (1.0 - c) * (1.0 - d)
                                j = c * (1.0 - d)
                                k = (1.0 - c) * d
                                l = c * d
                                u = (i * input[math.ceil(a)][math.ceil(b)] + j * input[math.ceil(a)][math.floor(b)] + k * input[math.floor(a)][math.ceil(b)] +  j * input[math.floor(a)][math.floor(b)] ) / (i + j + k + l)
                                output[x, y] = u

        

    cv2.imshow("Image",output)
    cv2.imwrite(outputFileName, output)
    cv2.waitKey()


def subtract():
    a = cv2.imread("face2.jpg", cv2.IMREAD_COLOR)
    b = cv2.imread("test5.jpg", cv2.IMREAD_COLOR)
    i = cv2.subtract(a, b)
    i = cv2.bitwise_not(i)
    cv2.imshow("Image", i)
    cv2.waitKey()


#problem1("face2.jpg", "test1.jpg", 21, 7, 2)
#problem2("face1.jpg", "test2.jpg", 0.9, 5, 2)
#problem3("face2.jpg", "test3.jpg")
#problem4("face2.jpg", "test4.jpg", 2, 150, 2, 0)
#problem4("test4.jpg", "test5.jpg", -0.3, 200, 1, 0)
#subtract()
