
import numpy as np


class GenNeirest:
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
       Neirest neighbor method: it select the image in videoSke that has the skeleton closest to the skeleton
    """
    def __init__(self, videoSkeTgt, debug=False):
        self.videoSkeletonTarget = videoSkeTgt
        self.debug=debug

    def generate(self, ske):           
        """ generator of image from skeleton """
        best_dist_near = float("+inf") # pour la distance => on s interesse nous à la plus petite distance entre les deux squeletes 
        best_ind_near = None # pour recup l indice de l image associé à la distance minimal trouver

        taille = self.videoSkeletonTarget.skeCount()
        if self.debug:
            print(f"nb squelette ds video cible  {taille}\n")

        # on parcourt notre dataset 
        for i in range(taille):
            ske_i = self.videoSkeletonTarget.ske[i] # on recup le squelette à l indice i 

            # on calcule la distance entre le squelette i cible et celui qu on veut 
            dist = ske_i.distance(ske)

            if self.debug and i<1:
                print(f" ske_i=[{i:3d}] => dist={dist:.4f}") 

            if dist < best_dist_near:
                best_dist_near = dist
                best_ind_near = i

        if self.debug:
            print("\n")
            if best_ind_near is not None:
                print(f"indice img = {best_ind_near} avec dist ={best_dist_near:.4f}")
                print(f"chemin img : {self.videoSkeletonTarget.imagePath(best_ind_near)}")
            else:
                print(" aucun res -> img blanche")
            print("\n")

        if best_ind_near is not None:
                return self.videoSkeletonTarget.readImage(best_ind_near)
        
        return np.ones((64,64, 3), dtype=np.uint8)

