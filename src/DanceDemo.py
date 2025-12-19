import numpy as np
import cv2
import sys

from VideoSkeleton import VideoSkeleton
from VideoSkeleton import combineTwoImages
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenNearest import GenNeirest
from GenVanillaNN import *
from GenGAN import *


class DanceDemo:
    """ class that run a demo of the dance.
        The animation/posture from self.source is applied to character define self.target using self.gen
    """
    def __init__(self, filename_src, typeOfGen=2):
        self.target = VideoSkeleton( "data/taichi1.mp4" )
        self.source = VideoReader(filename_src)
        self.ske_to_img = SkeToImageTransform(256) 
        if typeOfGen==1:           # Nearest
            print("Generator: GenNeirest")
            self.generator = GenNeirest(self.target, debug=False)
        elif typeOfGen==2:         # VanillaNN
            print("Generator: GenSimpleNN")
            self.generator = GenVanillaNN( self.target, loadFromFile=True, optSkeOrImage=1)
        elif typeOfGen==3:         # VanillaNN
            print("Generator: GenSimpleNN")
            self.generator = GenVanillaNN( self.target, loadFromFile=True, optSkeOrImage=2)
        elif typeOfGen==4:         # GAN ske
            print("Generator: GenSimpleNN")
            self.generator = GenGAN( self.target, loadFromFile=True,optSkeOrImage=1, gan_mode="bce")
        elif typeOfGen==5:         # GAN image
            print("Generator: GenSimpleNN")
            self.generator = GenGAN( self.target, loadFromFile=True, optSkeOrImage=2, gan_mode="bce")
        elif typeOfGen==6:         # GAN amélioré ske
            print("Generator: GenSimpleNN")
            self.generator = GenGAN( self.target, loadFromFile=True, optSkeOrImage=1, gan_mode="wgan-gp")
        elif typeOfGen==7:         # GAN amélioter image
            print("Generator: GenSimpleNN")
            self.generator = GenGAN( self.target, loadFromFile=True, optSkeOrImage=2, gan_mode="wgan-gp")
        else:
            print("DanceDemo: typeOfGen error!!!")

    def draw(self):
        ske = Skeleton()
        image_err = np.zeros((256, 256, 3), dtype=np.uint8)
        image_err[:, :] = (0, 0, 255)

        for i in range(self.source.getTotalFrames()):
            image_src = self.source.readFrame()

            if i % 5 == 0:
                isSke, image_src, ske = self.target.cropAndSke(image_src, ske)

                if isSke:
                    ske_img = self.ske_to_img(ske)
                    image_tgt = self.generator.generate(ske)
                    image_tgt = (image_tgt * 255).astype(np.uint8)
                else:
                    ske_img  = image_err.copy()
                    image_tgt = image_err.copy()

                H, W = 256, 256
                image_src = cv2.resize(image_src, (W, H))
                ske_img   = cv2.resize(ske_img,   (W, H))
                image_tgt = cv2.resize(image_tgt, (W, H))

                image_combined_un = combineTwoImages(image_src, ske_img)
                image_combined_deux = combineTwoImages(image_combined_un, image_tgt)

                image_combined = cv2.resize(image_combined_deux, (512, 256))
                cv2.imshow('Image', image_combined)

                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                if key & 0xFF == ord('n'):
                    self.source.readNFrames(100)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    # NEAREST = 1

    # VANILLA_NN_SKE = 2
    # VANILLA_NN_Image = 3

    # GAN_SKE = 4
    # GAN_Image = 5

    # GAN_W_KSE = 6
    # GAN_W_IMAGE = 7

    GEN_TYPE = 4
    OPT = 2

    if OPT == 1 : 
        ddemo = DanceDemo("data/taichi2_full.mp4", GEN_TYPE)
    elif OPT == 2 : 
        ddemo = DanceDemo("data/taichi2.mp4", GEN_TYPE)
    elif OPT == 3 : 
        ddemo = DanceDemo("data/karate1.mp4", GEN_TYPE)

    ddemo.draw()
