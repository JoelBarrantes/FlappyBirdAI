import numpy as np
import cv2
from scipy import misc


for i in range(0,999):
    im = cv2.imread('output/i-'+str(i)+'.png', 0)

    ret,thresh = cv2.threshold(im,51,255,0)
    #im2, contours, hierarchy = cv2.findCon cv2.FILLED)tours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #cv2.drawContours( im, contours, -1, (0,255,0), cv2.FILLED);
    #cv2.imshow( "Components",  thresh )


    misc.imsave('output/t2-'+str(i)+'.png', thresh)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
