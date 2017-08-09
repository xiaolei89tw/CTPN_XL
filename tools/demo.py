#
# The codes are used for implementing CTPN for scene text detection, described in: 
#
# Z. Tian, W. Huang, T. He, P. He and Y. Qiao: Detecting Text in Natural Image with
# Connectionist Text Proposal Network, ECCV, 2016.
#
# Online demo is available at: textdet.com
# 
# These demo codes (with our trained model) are for text-line detection (without 
# side-refiement part).  
#
#
# ====== Copyright by Zhi Tian, Weilin Huang, Tong He, Pan He and Yu Qiao==========

#            Email: zhi.tian@siat.ac.cn; wl.huang@siat.ac.cn
# 
#   Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences
#
#

from cfg import Config as cfg
from other import draw_boxes, resize_im, CaffeModel
import cv2, os, caffe, sys
from detectors import TextProposalDetector, TextDetector
import os.path as osp
from utils.timer import Timer
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

DEMO_IMAGE_DIR="Leads/"
TEXT_IMAGE_DIR="Texts/"
Detected_IMAGE_DIR="LeadsOut/"
#DEMO_IMAGE_DIR="demo_images/"
NET_DEF_FILE="models/deploy.prototxt"
MODEL_FILE="models/ctpn_trained_model.caffemodel"


def draw_0(im, bboxes, is_display=True, color=None, caption="Image", wait=False):
    """
        boxes: bounding boxes
    """
    im=im.copy()
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)

    for box in bboxes:
        if color==None:
            if len(box)==5 or len(box)==9:
                c=tuple(cm.jet([box[-1]])[0, 2::-1]*255)
            else:
                c=tuple(np.random.randint(0, 256, 3))
        else:
            c=color
        cv2.rectangle(im, tuple(box[:2]), tuple(box[2:4]), c)

    if is_display:
        cv2.imshow(caption, im)

    return im


def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)



def cutterimage(): 
    from PIL import Image
    i = Image.open('dt110507dhct.jpg')
    frame2 = i.crop(((275, 0, 528, 250)))


def img2text1(imagefile):

    from PIL import Image
    import pytesseract

    #pytesseract.pytesseract.tesseract_cmd = imagefile
    # Include the above line, if you don't have tesseract executable in your PATH
    # Example tesseract_cmd: 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'

    #print(pytesseract.image_to_string(Image.open(imagefile),  lang='eng')) 
    texts = pytesseract.image_to_string(Image.open(imagefile), lang = 'eng')
    #rint(pytesseract.image_to_string(Image.open(imagefile), lang='fra')) 

    return texts 


def img2text(imagefile):
    from tesserocr import PyTessBaseAPI, PSM, OEM


    with PyTessBaseAPI() as api:
        api.SetImageFile(imagefile)

        #os = api.DetectOrientationScript()
        #print ("Orientation: {orient_deg}\nOrientation confidence: {orient_conf}\n"
        #       "Script: {script_name}\nScript confidence: {script_conf}").format(**os)
        texts = api.GetUTF8Text() 
        #print api.AllWordConfidences()
        #print(texts)

    return texts 



def draw1(im, locclist, notes) : 
#[[  64.          228.40620422  783.          301.28256226    0.9761017 ]
#x0 y0 x1 y1 


    #im = np.array(Image.open(im_file), dtype=np.uint8)
    import PIL 
    # Create figure and axes
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)
    i = 1 
    # Create a Rectangle patch
    for locc in locclist: 
        d1 = int(locc[3] - locc[1])
        d0 = int(locc[2] - locc[0])
        rect = patches.Rectangle( ( int(locc[0]) , int(locc[1]) ),d0,d1,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        im2 = PIL.Image.fromarray(im)
        frame2 = im2.crop((int(locc[0]) , int(locc[1]), int(locc[2]) , int(locc[3])))  
        candname = TEXT_IMAGE_DIR + "Part" + str(i) + "from" + notes
        frame2.save(candname)
        texts = img2text(candname)
        #texts.encode('utf-8')
        t = texts.encode('ascii', 'ignore').decode('ascii')
        t = t.rstrip()
        import csv   
        fields = str(i) + '\t' + notes + '\t' + t 
        print fields

        with open('Texts.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([fields])
        i += 1 

    #plt.show()
    fig.savefig(Detected_IMAGE_DIR + notes + '.png')   # save the figure to file


    return True 


if len(sys.argv)>1 and sys.argv[1]=="--no-gpu":
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()
    caffe.set_device(cfg.TEST_GPU_ID)

# initialize the detectors
text_proposals_detector=TextProposalDetector(CaffeModel(NET_DEF_FILE, MODEL_FILE))
text_detector=TextDetector(text_proposals_detector)

demo_imnames=os.listdir(DEMO_IMAGE_DIR)
timer=Timer()

for im_name in demo_imnames:
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "Image: %s"%im_name

    im_file=osp.join(DEMO_IMAGE_DIR, im_name)
    im=cv2.imread(im_file)

    timer.tic()

    im, f=resize_im(im, cfg.SCALE, cfg.MAX_SCALE)
    text_lines=text_detector.detect(im)
    print(text_lines)
    print "Number of the detected text lines: %s"%len(text_lines)
    print "Time: %f"%timer.toc()
    locc = text_lines[0] 

    draw1(im, text_lines, im_name)
    #im_with_text_lines=draw_boxes(im, text_lines, caption=im_name, wait=False)

print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
print "Thank you for trying our demo. Press any key to exit..."
#cv2.waitKey(0)

