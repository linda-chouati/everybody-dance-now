import numpy as np
import cv2
import os
import sys
import copy 

from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
import torch 
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms

from VideoSkeleton import VideoSkeleton
from Skeleton import Skeleton

torch.set_default_dtype(torch.float32)


class SkeToImageTransform:
    def __init__(self, image_size):
        self.imsize = image_size

    def __call__(self, ske):
        #image = Image.new('RGB', (self.imsize, self.imsize), (255, 255, 255))
        image = white_image = np.ones((self.imsize, self.imsize, 3), dtype=np.uint8) * 255
        ske.draw(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('Image', image)
        # key = cv2.waitKey(-1)
        return image


class VideoSkeletonDataset(Dataset):
    def __init__(self, videoSke, ske_reduced, source_transform=None, target_transform=None):
        """ videoSkeleton dataset: 
                videoske(VideoSkeleton): video skeleton that associate a video and a skeleton for each frame
                ske_reduced(bool): use reduced skeleton (13 joints x 2 dim=26) or not (33 joints x 3 dim = 99)
        """
        self.videoSke = videoSke
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced
        print("VideoSkeletonDataset: ",
              "ske_reduced=", ske_reduced, "=(", Skeleton.reduced_dim, " or ",Skeleton.full_dim,")" )


    def __len__(self):
        return self.videoSke.skeCount()


    def __getitem__(self, idx):
        # prepreocess skeleton (input)
        ske = self.videoSke.ske[idx]
        ske = self.preprocessSkeleton(ske)
        # prepreocess image (output)
        image = Image.open(self.videoSke.imagePath(idx))
        if self.target_transform:
            image = self.target_transform(image)
        return ske, image

    
    def preprocessSkeleton(self, ske):
        if self.source_transform:
            ske = self.source_transform(ske)
        else:
            ske = torch.from_numpy( ske.__array__(reduced=self.ske_reduced).flatten() )
            ske = ske.to(torch.float32)
            ske = ske.reshape( ske.shape[0],1,1)
        return ske


    def tensor2image(self, normalized_image):
        numpy_image = normalized_image.detach().cpu().numpy()
        # Réorganiser les dimensions (C, H, W) en (H, W, C)
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        # passage a des images cv2 pour affichage
        numpy_image = cv2.cvtColor(np.array(numpy_image), cv2.COLOR_RGB2BGR)
        denormalized_image = numpy_image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
        denormalized_output = denormalized_image * 1
        return denormalized_output


def init_weights(m):
    """
    initialisaion manuel des poids 
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


########################################################################################
###################### APPROCHE 1 : à partir du vecteur squelette ######################
########################################################################################

class SEBlock1d(nn.Module):
    """
    pour mettre de l attention par canal 
    => pour aider le reseau a donner plus ou moins d importance à certaines feauture
    """
    def __init__(self, dim, reduction=8):
        """
        dim : nb features entrée
        reduc : facteur pour rduire la dimension
        """
        super().__init__()
        hidden = max(dim // reduction, 4) # au minim 4 neurones
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x): 
        """
        en gros le resau va apprendre un poids entre 0 et 1 pour chaque features
        - si importante -> poids renforcée
        - si pas bcp utile -> poids attenué
        """
        w = self.fc1(x)
        w = F.relu(w, inplace=True)
        w = self.fc2(w)
        w = torch.sigmoid(w)
        return x * w
    
class GenNNSke26ToImage(nn.Module): # ici c un reseau entreiere connecté 
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton_dim26)->Image
    """
    def __init__(self):
        super().__init__()
        self.input_dim = Skeleton.reduced_dim

        # bloc 1 
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256) # pour stabiliser l apprentissage
        self.se1 = SEBlock1d(256) # donc pour mettre de l attention 

        # bloc 2 
        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.se2 = SEBlock1d(512)

        # bloc 3 
        self.fc3 = nn.Linear(512, 1024)
        self.bn3 = nn.BatchNorm1d(1024)

        # sortie 
        self.fc_out = nn.Linear(1024, 3 * 64 * 64)

        # on ajoute des skips connection 
        # pour garder de l info des couches d avant 
        self.skip1 = nn.Linear(256, 512)
        self.skip2 = nn.Linear(512, 1024)

    def forward(self, z):
        z = z.view(z.size(0), -1)  

        # passe avant dans le bloc 1 et ainsi de suite 
        h1 = F.leaky_relu(self.bn1(self.fc1(z)), 0.2)
        h1 = self.se1(h1)

        h2 = F.leaky_relu(self.bn2(self.fc2(h1)), 0.2)
        h2 = self.se2(h2)
        h2 = h2 + self.skip1(h1) 

        h3 = F.leaky_relu(self.bn3(self.fc3(h2)), 0.2)
        h3 = h3 + self.skip2(h2) 

        out = torch.tanh(self.fc_out(h3))
        return out.view(z.size(0), 3, 64, 64)



###########################################################################################
###################### APPROCHE 2 : à partir de l image du squelette ######################
###########################################################################################


class ResBlock2d(nn.Module):
    """
    l idee ici n est pas de reconstuire l image complete 
    mais d apprendre quel parties doivent etre modifié
    -> donc reseau apprends une petite corerection à ajouter à l entrée 
    """
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        out = self.block(x)
        out = out + x      
        return F.relu(out, inplace=True)


class SelfAttention2d(nn.Module):
    """
    en gros chaque pixel pourra prendre en compte les autres de toutes l image
    et non plus seulement c est voisins 
    """
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1) # ce qu un pix cherche quoi important pour lui 
        self.key_conv   = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1) # avec quoi on compare 
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1) # info à propager
        self.gamma = nn.Parameter(torch.zeros(1)) # param pour controler l importance de l attention pour qu il soit vraimen utile 

    def forward(self, x):
        # mise en forme pour cimparer chaque pos avec toutes les autres
        B, C, H, W = x.size()
        N = H * W
        proj_query = self.query_conv(x).view(B, -1, N).permute(0, 2, 1)
        proj_key   = self.key_conv(x).view(B, -1, N)
        # on calcul la similitude entre les pos
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        # propagation de l info pondere par l attention 
        proj_value = self.value_conv(x).view(B, C, N)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x   # on ajoute aussi l entré pour garder l info initial


class GenNNSkeImToImage(nn.Module):
    """
    meme principe que le tp de Julie Digne avec une archi "u-net"
    -> donc un aucodeur-decodeur avec skip connection 
    """
    def __init__(self):
        """ class that Generate a new image from from THE IMAGE OF the new skeleton posture
        SkeletonImage is an image with the skeleton drawed on it
        Fonc generator(SkeletonImage)->Image
        """
        super().__init__()
        
        # Partie ENCODEUR : reduction taille, extraire features
        self.enc1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)      # 64 x 32 x 32

        self.enc2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)    # 128 x 16 x 16
        self.bn2  = nn.BatchNorm2d(128)

        self.enc3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)   # 256 x 8 x 8
        self.bn3  = nn.BatchNorm2d(256)

        self.enc4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)   # 512 x 4 x 4
        self.bn4  = nn.BatchNorm2d(512)

        # en plus : l attention + 2 blocs résiduel
        self.attn = SelfAttention2d(512)
        self.res1 = ResBlock2d(512)
        self.res2 = ResBlock2d(512)

        # partie DECODEUR : recup detail -> reconstuiction de l img
        self.dec1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.dbn1 = nn.BatchNorm2d(256)
        self.dec1_conv = nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1)   # on concat avec e3

        self.dec2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dec2_conv = nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1)   #  avec e2

        self.dec3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dbn3 = nn.BatchNorm2d(64)
        self.dec3_conv = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)      #  avec e1

        self.dec4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        e1 = F.relu(self.enc1(z))               

        e2 = F.relu(self.bn2(self.enc2(e1)))     
        e3 = F.relu(self.bn3(self.enc3(e2)))   
        e4 = F.relu(self.bn4(self.enc4(e3)))    

        b  = self.attn(e4)
        b  = self.res1(b)
        b  = self.res2(b)

        d1 = F.relu(self.dbn1(self.dec1(b)))     
        d1 = torch.cat([d1, e3], dim=1)         
        d1 = F.relu(self.dec1_conv(d1))        

        d2 = F.relu(self.dbn2(self.dec2(d1)))   
        d2 = torch.cat([d2, e2], dim=1)         
        d2 = F.relu(self.dec2_conv(d2))       

        d3 = F.relu(self.dbn3(self.dec3(d2)))   
        d3 = torch.cat([d3, e1], dim=1)         
        d3 = F.relu(self.dec3_conv(d3))         

        out = torch.tanh(self.dec4(d3))         
        return out

##############################################################################
##############################################################################

class GenVanillaNN():
    """ class that Generate a new image from a new skeleton posture
        Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        image_size = 64

        if optSkeOrImage==1:        # skeleton_dim26 to image
            self.netG = GenNNSke26ToImage()
            self.netG.apply(init_weights)
            src_transform = None 
            self.filename = 'data/Dance/DanceGenVanillaFromSke26.pth'
        else:                       # skeleton_image to image
            self.netG = GenNNSkeImToImage()
            src_transform = transforms.Compose([ SkeToImageTransform(image_size), # c ici qu on cree l image en stick 
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 ])
            self.filename = 'data/Dance/DanceGenVanillaFromSkeim.pth'

        tgt_transform = transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            # [transforms.Resize((64, 64)),
                            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
                            # ouput image (target) are in the range [-1,1] after normalization
        
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform, source_transform=src_transform)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.netG.to(device)

        if loadFromFile and os.path.isfile(self.filename):
            print("GenVanillaNN: Load=", self.filename)
            print("GenVanillaNN: Current Working Directory: ", os.getcwd())
            state_dict = torch.load(self.filename, map_location=device)
            self.netG.load_state_dict(state_dict)


    def train(self, n_epochs=20, patience=5, min_delta=0.0, debug=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.netG.to(device)

        criterion = nn.L1Loss() # apparement mieux que mse pour l image
        optimiser = torch.optim.Adam(self.netG.parameters(), lr=0.001, betas=(0.5, 0.999))

        dataset_size = len(self.dataset)
        val_size = int(0.2 * dataset_size) # 20pourcent pour la validation 
        train_size = dataset_size - val_size

        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])

        train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)

        if debug:
            print(f"dataset total : {dataset_size} => pour train = {train_size} et pour val = {val_size}")

        # pour faire l early stopping 
        best_val_loss = float('inf')
        best_model = copy.deepcopy(self.netG.state_dict())
        epoch_with_no_improvement = 0

        for epoch in range(n_epochs): 
            self.netG.train()
            epoch_loss = 0.0
            nb_bacthes = 0

            for _, (ske_bacth, img_batch) in enumerate(train_loader): 
                ske_bacth = ske_bacth.to(device) # inputs 
                img_batch = img_batch.to(device) # target

                optimiser.zero_grad()
                
                # passe avant 
                img_pred = self.netG(ske_bacth)

                # calcule perte 
                loss = criterion(img_pred, img_batch)

                # passe arriere avec uptade des poids
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()
                nb_bacthes += 1

            # moy de la perte sur cette epoch
            avg_loss = epoch_loss / max(nb_bacthes, 1)

            # partie pour la validation 
            self.netG.eval()
            epoch_val_loss = 0.0
            nb_bacthes_val = 0

            with torch.no_grad():
                for ske_batch, img_batch in val_loader:
                    ske_batch = ske_batch.to(device)
                    img_batch = img_batch.to(device)

                    img_pred = self.netG(ske_batch)
                    val_loss = criterion(img_pred, img_batch)

                    epoch_val_loss += val_loss.item()
                    nb_bacthes_val += 1
                
            avg_val_loss = epoch_val_loss / max(nb_bacthes_val, 1)
            print(f"=> Epoch {epoch+1}/{n_epochs} - train_loss = {avg_loss:.4f} | val_loss = {avg_val_loss:.4f}")

            # on verifie si on arrete ou pas l entrainement 
            if avg_val_loss + min_delta < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = copy.deepcopy(self.netG.state_dict())
                epoch_with_no_improvement = 0
                if debug:
                    print(f" nouvelle loss pour la val : {best_val_loss:.4f}")
            else:
                epoch_with_no_improvement += 1
                if debug:
                    print(f"pas d amélioration significative (epochs_no_improve = {epoch_with_no_improvement}/{patience})")

                if epoch_with_no_improvement >= patience:
                    if debug:
                        print("arret de l entrianement")
                    break


        # on recup les meilleur poids pour la sauvegarde
        self.netG.load_state_dict(best_model)
        torch.save(self.netG.state_dict(), self.filename)
        print(f"===> train termine, model save dans : {self.filename}")



    def generate(self, ske):
        """ generator of image from skeleton """
        self.netG.eval()
        device = next(self.netG.parameters()).device

        ske_t = self.dataset.preprocessSkeleton(ske) # donne le squellete en un tensor 
        ske_t_batch = ske_t.unsqueeze(0).to(device)        # make a batch

        with torch.no_grad():
            normalized_output = self.netG(ske_t_batch) # passe dans le reseaux
        
        res = self.dataset.tensor2image(normalized_output[0])       # get image 0 from the batch
        return res



##############################################################################
##############################################################################

if __name__ == '__main__':
    force = False
    optSkeOrImage = 2           # use as input a skeleton (1) or an image with a skeleton drawed (2)
    n_epoch = 500 
    train = True

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
    print("GenVanillaNN: Current Working Directory=", os.getcwd())
    print("GenVanillaNN: Filename=", filename)
    print("GenVanillaNN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    if train: 
        # Train
        gen = GenVanillaNN(targetVideoSke,optSkeOrImage=optSkeOrImage, loadFromFile=False)
        gen.train(n_epoch, patience=7, min_delta=1e-4, debug=False)
    else:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=True)    # load from file        


    # Test with a second video
    for i in range(targetVideoSke.skeCount()):
        image = gen.generate( targetVideoSke.ske[i] )
        #image = image*255
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)

        cv2.imshow('Image', image)

        key = cv2.waitKey(-1)
        if key == ord('q'): 
            break

