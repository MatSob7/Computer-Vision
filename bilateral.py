import cv2
import numpy

def bilateral( img_in, sigma_s, sigma_v, reg_constant=1e-8 ):

    #   img_in       (ndarray) monochrome input image
    #   sigma_s      (float)   spatial gaussian std. dev.
    #   sigma_v      (float)   value gaussian std. dev.
    #   reg_constant (float)   optional regularization constant for pathalogical cases


    gauss = lambda r2, sigma: (numpy.exp( -0.5*r2/sigma**2 )*3).astype(int)*1.0/3.0

    width = int( 3*sigma_s+1 )

    wsum = numpy.ones( img_in.shape )*reg_constant
    result  = img_in*reg_constant

    for shift_x in range(-width,width+1):
        for shift_y in range(-width,width+1):

            w = gauss( shift_x**2+shift_y**2, sigma_s )

            off = numpy.roll(img_in, [shift_y, shift_x], axis=[0,1] )

            fin = w*gauss( (off-img_in)**2, sigma_v )

            result += off*fin
            wsum += fin

    return result/wsum


I = cv2.imread('images/lena.png', cv2.IMREAD_UNCHANGED ).astype(numpy.float32)/255.0

B = numpy.stack([
        filter_bilateral( I[:,:,0], 10.0, 0.1 ),
        filter_bilateral( I[:,:,1], 10.0, 0.1 ),
        filter_bilateral( I[:,:,2], 10.0, 0.1 )], axis=2 )

O = numpy.hstack( [I,B] )

cv2.imwrite( 'images/lena_bilateral.png', out*255.0 )
