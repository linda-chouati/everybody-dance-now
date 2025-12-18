
import cv2
import os
import sys
import numpy as np 

import torch.nn as nn
import torch
from torchvision import transforms

from VideoSkeleton import VideoSkeleton
from GenVanillaNN import * 


#################################################################################
############################# DICSRIMINATOR #####################################
#################################################################################


class Discriminator(nn.Module):
    """
    le discriminator : reseau (CNN) qui recoit l image et qui juge si vrai ou fausse image 
    - gan classique (bce) : score = proba que l image soit vrai
    - wgan-gp : score = qalité / réaliste (donc pas une proba)
    """
    def __init__(self, ngpu=0, gan_mode="bce"):
        super().__init__()
        self.ngpu = ngpu # comprends pas à quoi ca sert 

        # pour le wgan gp : on utilise intanceNorm2d pour pas fausser le caulcul de penalité des gradients
        if gan_mode == "wgan-gp":
            NormLayer = nn.InstanceNorm2d # normalise image par image 
        else:
            NormLayer = nn.BatchNorm2d # normalise tout le bacth 

        self.model = nn.Sequential(
            # bloc 1 
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # bloc 2 
            nn.Conv2d(64, 128, 4, 2, 1),
            NormLayer(128),
            nn.LeakyReLU(0.2, inplace=True),

            # bloc 2
            nn.Conv2d(128, 256, 4, 2, 1),
            NormLayer(256),
            nn.LeakyReLU(0.2, inplace=True),

            # bloc 4 
            nn.Conv2d(256, 512, 4, 2, 1),
            NormLayer(512),
            nn.LeakyReLU(0.2, inplace=True),

            # dernire couche => le score 
            nn.Conv2d(512, 1, 4, 1, 0)
        )

    def forward(self, x):
        out = self.model(x)  
        return out.view(-1)  

#################################################################################
############################# GAN #####################################
#################################################################################


class GenGAN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1, gan_mode="bce"):
        self.optSkeOrImage = optSkeOrImage
        self.gan_mode = gan_mode # bce ou wgan-gp
        image_size = 64

        # config du generator -> on recup ceux deja codé avant 
        if gan_mode == "bce":  # gan classique 
            if optSkeOrImage == 1: #  strat avec squelette reduce
                self.netG = GenNNSke26ToImage() 
                src_transform = None      
                self.filename = 'data/Dance/DanceGenGAN_bce_fromSke26.pth'
            else: # strat avec image en baton du sque
                self.netG = GenNNSkeImToImage()
                src_transform = transforms.Compose([
                    SkeToImageTransform(image_size),      
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5),
                                        (0.5, 0.5, 0.5)),
                ])
                self.filename = 'data/Dance/DanceGenGAN_bce_fromSkeIm.pth'
        else: # version gan ameliorer 
            if optSkeOrImage == 1:
                self.netG = GenNNSke26ToImage()
                src_transform = None     
                self.filename = 'data/Dance/DanceGenGAN_wgan_fromSke26.pth'
            else:
                self.netG = GenNNSkeImToImage()
                src_transform = transforms.Compose([
                    SkeToImageTransform(image_size),           # dessine le squelette sur une image
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5),
                                        (0.5, 0.5, 0.5)),
                ])
                self.filename = 'data/Dance/DanceGenGAN_wgan_fromSkeIm.pth'

        # transformer pour la cilbe video
        tgt_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)),
        ])

        self.dataset = VideoSkeletonDataset(
            videoSke,
            ske_reduced=True,
            source_transform=src_transform,
            target_transform=tgt_transform
        )

        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=64,
            shuffle=True
        )

        # device à changer aussi dans les autres fichier pour le mps pour mon mac  
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"DEVISE : {self.device}")

        self.netG.to(self.device)
        self.netD = Discriminator(gan_mode=gan_mode).to(self.device)

        # init pois
        self.netG.apply(init_weights)
        self.netD.apply(init_weights)

        # pour charger un modele deja existant 
        if loadFromFile and os.path.isfile(self.filename):
            print("GenGAN: Load=", self.filename, "   Current Working Directory=", os.getcwd())
            state_dict = torch.load(self.filename, map_location=self.device)
            self.netG.load_state_dict(state_dict)

        self.real_label = 1.0 # label pour quand le discriminator va predire si vrai ou fake image 
        self.fake_label = 0.0

    # pour le wgan gpe 
    def compute_gradient_penalty(self, real_samples, fake_samples):
            """Compute gradient penalty for WGAN-GP"""
            device = real_samples.device
            batch_size = real_samples.size(0)
            alpha = torch.rand(batch_size, 1, 1, 1, device=device)
            alpha = alpha.expand_as(real_samples)

            # Interpolate between real and fake samples
            interpolated = alpha * real_samples + (1 - alpha) * fake_samples
            interpolated.requires_grad_(True)

            # Get discriminator output for interpolated images
            d_interpolated = self.netD(interpolated)

            # Calculate gradients of probabilities with respect to examples
            gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated,
                                            grad_outputs=torch.ones_like(d_interpolated),
                                            create_graph=True, retain_graph=True)[0]

            # Calculate gradient penalty
            gradients = gradients.view(batch_size, -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

            return gradient_penalty

    def train(self, n_epochs=20):
        if self.gan_mode == "bce":
            return self.train_bce(n_epochs)
        elif self.gan_mode == "wgan-gp":
            return self.train_wgan_gp(n_epochs)
        else:
            print("Methode pas connu pour partie gan")

    ###############################################################
    def train_bce(self, n_epochs):
        """
        pour entrainer un gan classique :
        - discriminateur : apprends à distinger une igame vrai d une image genere par le génrateur 
        - générator : apprends à produire une image réaliste et proche de la vraie image
        ppour la perte : 
        - perte adverserial 
        - perte de reconstuction (L1)

        """
        criterion = nn.BCEWithLogitsLoss() # pour le discriminateur 
        l1_loss = nn.L1Loss() # classique comme avant (force l image generé à ressebmler à la vraie)
        lambda_l1 = 50 # poids pour forcer la ressemblance visuelle

        lr = 0.0002
        # du coup deux optimisateur separer
        optimizerD = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizerG = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(0.5, 0.999))

        for epoch in range(n_epochs):
            running_loss_D, running_loss_G = 0.0, 0.0
            
            for _, (ske_batch, real_batch) in enumerate(self.dataloader):
                ske_batch = ske_batch.to(self.device) # entrée du generateu 
                real_batch = real_batch.to(self.device) # image réelle 
                b_size = real_batch.size(0)

                # label avec lissage pour ectier que le discriminator soit trop confiant 
                real_label = torch.full((b_size,), 0.9, device=self.device)
                fake_label = torch.full((b_size,), 0.0, device=self.device)

                    # 1 entrainement du discriminator 
                self.netD.zero_grad()

                # cas 1 : images réelles -> D doit prédire vrai : 1 
                loss_real = criterion(self.netD(real_batch), real_label)
                
                # cas 2 : images généré 
                fake_img = self.netG(ske_batch) # donc bien img qui vient du générator 
                loss_fake = criterion(self.netD(fake_img.detach()), fake_label) # ici D doit trouvé 0 
                
                # perte total du discriminator 
                loss_D = loss_real + loss_fake
                # dans la passe arriere but : mieux reconnaitre les vraies img / mieux rejeter celles généré 
                loss_D.backward()
                optimizerD.step()

                # 2 entrainement du générator -> on fige le discriminator 
                self.netG.zero_grad()
                pred_fake = self.netD(fake_img)
                loss_G_gan = criterion(pred_fake, real_label) # pour le realisme
                loss_G_l1 = l1_loss(fake_img, real_batch) # pour la fidelité visuelle 
                
                # perte total du generotor 
                loss_G = loss_G_gan + lambda_l1 * loss_G_l1
                loss_G.backward()
                optimizerG.step()

                running_loss_D += loss_D.item()
                running_loss_G += loss_G.item()

            print(f"[Epoch {epoch+1}/{n_epochs}] Loss_D: {running_loss_D/len(self.dataloader):.4f} | Loss_G: {running_loss_G/len(self.dataloader):.4f}")

        torch.save(self.netG.state_dict(), self.filename)
        print("Sauvegarde terminé.")


    ###############################################################
    def train_wgan_gp(self, n_epochs):
        l1_loss = nn.L1Loss()
        
        # Hyperparamètres spécifiques WGAN
        lr = 1e-4
        lambda_gp = 10.0 # poids de la pénalité de graduit 
        lambda_l1 = 50.0 # pareil ici pour forcer la ressemblance visuelle 
        n_critic = 5 # ici le "discriminateur" est entrainé plus souvent que le générateur 

        optimizerD = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(0.0, 0.9))
        optimizerG = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(0.0, 0.9))

        step = 0
        for epoch in range(n_epochs):
            running_loss_D = 0.0
            running_loss_G = 0.0
            g_update_count = 0 # pour calculer la moyenne correcte

            for ske_batch, real_batch in self.dataloader:
                ske_batch = ske_batch.to(self.device)
                real_batch = real_batch.to(self.device)

                # 1 entrianement du critique "discriminateur"
                self.netD.zero_grad()
                
                # generation d imae fake
                fake_images = self.netG(ske_batch).detach()
                # score du critic
                d_real = self.netD(real_batch).mean() # celui la doit etre grand 
                d_fake = self.netD(fake_images).mean() # et lui petit (à reverif)
                # penalité du gradient 
                gp = self.compute_gradient_penalty(real_batch, fake_images)
                
                # perte du critic 
                loss_D = d_fake - d_real + lambda_gp * gp
                loss_D.backward()
                optimizerD.step()
                running_loss_D += loss_D.item()

                # 2 entrainement du generator (donc c ici qu on le fait moins svt que le critic)
                if step % n_critic == 0:
                    self.netG.zero_grad()
                    fake_images = self.netG(ske_batch)
                    d_fake = self.netD(fake_images).mean()
                    
                    loss_G_adv = -d_fake
                    loss_G_l1 = l1_loss(fake_images, real_batch)
                    
                    loss_G = loss_G_adv + lambda_l1 * loss_G_l1
                    loss_G.backward()
                    optimizerG.step()
                    
                    running_loss_G += loss_G.item()
                    g_update_count += 1
                
                step += 1

            avg_loss_D = running_loss_D / len(self.dataloader)
            avg_loss_G = running_loss_G / max(1, g_update_count)
            print(f"[WGAN Epoch {epoch+1}/{n_epochs}] Loss_D: {avg_loss_D:.4f} | Loss_G: {avg_loss_G:.4f}")

        torch.save(self.netG.state_dict(), self.filename)
        print("Sauvegarde terminée.")

    ###############################################################
    def generate(self, ske):
        self.netG.eval()
        ske_t = self.dataset.preprocessSkeleton(ske)
        ske_t_batch = ske_t.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            normalized_output = self.netG(ske_t_batch)
        
        img = self.dataset.tensor2image(normalized_output[0])
        return img


##############################################################################
##############################################################################

if __name__ == '__main__':
    force = False
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    gan_mode = "wgan-gp" #  bce ou wgan-gp
    optSkeOrImage = 2   # 1 = vecteur squelette, 2 = image stick

    gen = GenGAN(targetVideoSke, loadFromFile=False, optSkeOrImage=optSkeOrImage, gan_mode=gan_mode)
    gen.train(200)

    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        image = (image * 255).astype(np.uint8)
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break 

    cv2.destroyAllWindows()

