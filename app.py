import os
import sys
import cv2
import numpy as np
import streamlit as st
import tempfile
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_OK = True
except Exception:
    MOVIEPY_OK = False


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenNearest import GenNeirest
from GenVanillaNN import GenVanillaNN
from GenGAN import GenGAN


GEN_LABELS = {
    1: "Nearest",
    2: "VanillaNN – squelette 26D",
    3: "VanillaNN – image stick",
    4: "GAN (BCE) – squelette 26D",
    5: "GAN (BCE) – image stick",
    6: "WGAN-GP – squelette 26D",
    7: "WGAN-GP – image stick",
}

# ------------------------------------------------------------------
#  construction du générateur 
# ------------------------------------------------------------------

def build_generator(target_video, gen_type):
    """
    gen_type (tous ceux qu on a codé) :
        1 -> nearest
        2 -> vanillaNN (input = ske 26D)
        3 -> vanillaNN (input = image stick)
        4 -> GAN (input = ske 26D)
        5 -> GAN (input = image stick)
    """
    if gen_type == 1:
        return GenNeirest(target_video, debug=False)

    if gen_type == 2:
        return GenVanillaNN(target_video, loadFromFile=True, optSkeOrImage=1)

    if gen_type == 3:
        return GenVanillaNN(target_video, loadFromFile=True, optSkeOrImage=2)

    if gen_type == 4:
        return GenGAN(target_video, loadFromFile=True, optSkeOrImage=1, gan_mode="bce")

    if gen_type == 5:
        return GenGAN(target_video, loadFromFile=True, optSkeOrImage=2, gan_mode="bce")

    if gen_type == 6:
        return GenGAN(target_video, loadFromFile=True, optSkeOrImage=1, gan_mode="wgan-gp")

    if gen_type == 7:
        return GenGAN(target_video, loadFromFile=True, optSkeOrImage=2, gan_mode="wgan-gp")

    raise ValueError("type de generateur inconnu")


# ------------------------------------------------------------------
#  pour generer la video 
# ------------------------------------------------------------------

def add_audio_to_video(silent_video_path, src_video_path):
    """
    pour ajouter le son à une video lors de la démo 
    """
    if not MOVIEPY_OK:
        return silent_video_path

    video_clip = VideoFileClip(silent_video_path)
    src_clip = VideoFileClip(src_video_path)

    final_clip = video_clip.set_audio(src_clip.audio)

    tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_with_audio = tmp_file.name
    tmp_file.close()

    final_clip.write_videofile(
        out_with_audio,
        codec="libx264",
        audio_codec="aac",
        verbose=False,
        logger=None,
    )

    video_clip.close()
    src_clip.close()
    final_clip.close()

    #return out_with_audio
    return None

def make_dance_demo_video(src_video_path, gen_type):
    """
    retourne le chemin de la vidéo temporaire à affiché
    """

    target_vs = VideoSkeleton("data/taichi1.mp4")
    source = VideoReader(src_video_path)
    generator = build_generator(target_vs, gen_type)

    ske = Skeleton()
    image_err = np.zeros((256, 256, 3), dtype=np.uint8)
    image_err[:, :] = (0, 0, 255)
    TARGET_HEIGHT = 800

    video_writer = None
    out_path = None

    total_frames = int(source.getTotalFrames())
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(total_frames):
        image_src_raw = source.readFrame()
        if image_src_raw is None:
            break

        if i % 5 != 0:
            continue

        image_for_ske_detection = image_src_raw.copy()
        isSke, image_cropped_real, ske = target_vs.cropAndSke(
            image_for_ske_detection, ske
        )

        if isSke:
            h_crop, w_crop = image_cropped_real.shape[:2]
            image_skeleton_only = np.zeros((h_crop, w_crop, 3), dtype=np.uint8)
            ske.draw(image_skeleton_only)
            image_middle_viz = image_skeleton_only

            image_generated = generator.generate(ske)
            if image_generated.max() <= 1.5:
                image_generated = (image_generated * 255.0).clip(0, 255).astype(
                    np.uint8
                )
            else:
                image_generated = image_generated.clip(0, 255).astype(np.uint8)
        else:
            image_middle_viz = image_err
            image_generated = image_err

        # redimensionnement mais pas l air de mieux 
        h1, w1 = image_src_raw.shape[:2]
        new_w1 = int(w1 * (TARGET_HEIGHT / h1))
        img_view_1 = cv2.resize(image_src_raw, (new_w1, TARGET_HEIGHT))

        h2, w2 = image_middle_viz.shape[:2]
        new_w2 = int(w2 * (TARGET_HEIGHT / h2))
        img_view_2 = cv2.resize(image_middle_viz, (new_w2, TARGET_HEIGHT))

        img_view_3 = cv2.resize(image_generated, (TARGET_HEIGHT, TARGET_HEIGHT))

        image_combined = np.hstack((img_view_1, img_view_2, img_view_3))

        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            fps = max(1.0, float(source.getVideoFps()) / 5.0)

            tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            out_path = tmp_file.name
            tmp_file.close()

            video_writer = cv2.VideoWriter(
                out_path,
                fourcc,
                fps,
                (image_combined.shape[1], image_combined.shape[0]),
            )

        video_writer.write(image_combined)

        progress_bar.progress((i + 1) / total_frames)
        status_text.text(f"Traitement des frames... ({i+1}/{total_frames})")

    source.release()
    progress_bar.empty()
    status_text.empty()
    
    if video_writer is not None:
        video_writer.release()
        out_with_audio = add_audio_to_video(out_path, src_video_path)
        return out_with_audio
    else:
        raise RuntimeError("Aucune frame valide n'a été généré.")



# ------------------------------------------------------------------
#  parti pour l interface stramlit 
# ------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Dance Demo", layout="wide")

    # ------------------ un peu de style pour les marges ------------------
    st.markdown("""
        <style>
            .stSidebar > div { padding-top: 20px; }
            .block-container { padding-top: 2rem; }
            .element-container { margin-top: 20px; margin-bottom: 20px; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>Clone My Moves</h1>", unsafe_allow_html=True)

    # ------------------ choix vidéo source ------------------
    video_options = {
        "Taichi 2 (court)": "data/taichi2.mp4",
        "Taichi 2 (full)": "data/taichi2_full.mp4",
        "Karate 1": "data/karate1.mp4",
    }
    video_label = st.sidebar.selectbox(
        "Choisir une vidéo :", list(video_options.keys())
    )
    src_video_path = video_options[video_label]

    uploaded = st.sidebar.file_uploader(
        "Ou charger une autre vidéo (.mp4)", type=["mp4"]
    )
    if uploaded is not None: 
        user_video_path = os.path.join("data", "user_video_app.mp4")
        with open(user_video_path, "wb") as f:
            f.write(uploaded.read())
        src_video_path = user_video_path

    # ------------------ choix générateur ------------------
    GEN_OPTIONS = {
        "Nearest": 1,
        "VanillaNN – squelette 26D": 2,
        "VanillaNN – image stick": 3,
        "GAN (BCE) – squelette 26D": 4,
        "GAN (BCE) – image stick": 5,
        "WGAN-GP – squelette 26D": 6,
        "WGAN-GP – image stick": 7,
    }

    gen_type_label = st.sidebar.selectbox(
        "Type de générateur :",
        list(GEN_OPTIONS.keys()),
        index=2,
    )

    gen_type = GEN_OPTIONS[gen_type_label]

    # ------------------ bouton pour generer la vidio ------------------
    generate_clicked = st.sidebar.button(
        "Générer la vidéo final (source + squelette + génération)"
    )

    # ------------------ zone principale ------------------
    if generate_clicked:
        with st.spinner("Génération de la vidéo, ça peut prendre un peu de temps..."):
            out_path = make_dance_demo_video(src_video_path, gen_type)

        with open(out_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)

        st.markdown(
            f"Vidéo composée à partir de : `{src_video_path}`  et du generateur : **{GEN_LABELS[gen_type]}**"
        )

    else:
        st.info("Choisit une vidéo et un générateur puis clique sur le bouton dans la barre de gauche pour la génération.")



if __name__ == "__main__":
    main()
