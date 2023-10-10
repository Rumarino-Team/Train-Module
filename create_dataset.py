import pytube 
import os
import subprocess
import cv2
import numpy as np
import random
import argparse
import torch
from torchvision import transforms
# from albumentations import Compose, Rotate, Resize, ShiftScaleRotate
from albumentations import (
    Compose, Rotate, Resize, ShiftScaleRotate, OneOf, 
    IAAAdditiveGaussianNoise, GaussNoise, 
    MotionBlur, MedianBlur, Blur, 
    OpticalDistortion, GridDistortion, IAAPiecewiseAffine,
    HueSaturationValue, CLAHE, Perspective, RandomScale
)

def getVideoFramesFromYoutube(video_url: str, output_dir: str, output_format: str, fps: int):
    # Create a YouTube object with the URL
    yt = pytube.YouTube(video_url)
    # Get the video streams
    streams = yt.streams
    # Filter for video streams with a resolution of 720p
    video_stream = streams.filter(resolution="720p").first()
    # Download the video stream
    video_stream.download()
    # Get the name of the downloaded video file
    video_file_name = video_stream.default_filename
    # Create a directory to store the pngs
    png_dir = os.path.join(os.getcwd(), output_dir)
    # If the directory does not exist, create it
    if not os.path.exists(png_dir):
        os.mkdir(png_dir)
    # Convert the video file to pngs
    subprocess.run([
        "ffmpeg", "-i", video_file_name,
        "-vf", f"fps={fps}",
        os.path.join(png_dir, f"%03d.{output_format}")
    ])


def getVideoFrames(video_path: str, output_dir: str, output_format: str, fps: int):
    # Create a directory to store the pngs
    png_dir = os.path.join(os.getcwd(), output_dir)
    # If the directory does not exist, create it
    if not os.path.exists(png_dir):
        os.mkdir(png_dir)
    # Convert the video file to pngs
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        os.path.join(png_dir, f"%03d.{output_format}")
    ])

def is_collision(new_box, existing_boxes):
    for existing_box in existing_boxes:
        if (new_box[0] < existing_box[2] and
            new_box[2] > existing_box[0] and
            new_box[1] < existing_box[3] and
            new_box[3] > existing_box[1]):
            return True
    return False



def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Unchanged to keep the alpha channel
    return image  # Convert from BGRA to RGBA

def overlay_images(background, foreground, x, y):
    # We assume the foreground image is RGBA and the background is RGB.
    fg_img = foreground[..., :3]
    fg_alpha = foreground[..., 3]/255

    # Apply the foreground's alpha channel to its RGB channels.
    fg_alpha_expanded = np.expand_dims(fg_alpha, axis=2)
    fg_img_alpha = fg_alpha_expanded * fg_img

    background = background[..., :3]
    # Expand the alpha channel for the background
    bg_alpha_expanded = np.expand_dims(1 - fg_alpha, axis=2)
    bg_img_alpha = bg_alpha_expanded * background[y:y+fg_img.shape[0], x:x+fg_img.shape[1], :3]  # Corrected this line

    # Add the alpha-weighted foreground and background.
    overlaid_section = fg_img_alpha + bg_img_alpha

    # Replace original with overlay
    result = background.copy()
    result[y:y+fg_img.shape[0], x:x+fg_img.shape[1]] = overlaid_section

    return result



def data_augmentation_pipeline():
    return Compose([
        Rotate(limit=45, p=0.9, border_mode=cv2.BORDER_CONSTANT),
        Resize(100, 100),
        RandomScale(scale_limit=0.5, interpolation=cv2.INTER_LINEAR, p=0.5),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75, border_mode=cv2.BORDER_CONSTANT),
        OneOf([
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
        ], p=0.2),
                OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            Perspective(scale=(0.05, 0.1)),  # Adding Perspective transformation
        ], p=0.2),
        # OneOf([
        #     HueSaturationValue(10,15,10),
        #     CLAHE(clip_limit=2),         
        # ], p=0.3),
        
    ])

def createDataset(output_dir: str, output_format: str, object_path: str):
    # Ensure dataset directory exists, if not, create it
    os.makedirs(output_dir, exist_ok=True)

    # List all files in directory
    files = os.listdir(output_dir)
    object_files = os.listdir(object_path)

    # Filter out all non-png files
    background_files = [i for i in files if i.endswith(f'.{output_format}')]
    object_png_files = [i for i in object_files if i.endswith(f'.{output_format}')]

    # Assign each object a unique class ID
    object_class_ids = {object_file: i for i, object_file in enumerate(object_png_files)}

    # Sort files alphabetically to maintain a consistent order
    background_files.sort()
    object_png_files.sort()
    transform = data_augmentation_pipeline()

    # Loop over your background images
    for i, file in enumerate(background_files):
        # Load the background image
        background_img = cv2.imread(os.path.join(output_dir, file))

        # Number of objects to be placed on each background image
        num_objects = random.randint(0, 5)

        # List to store bounding boxes for each image
        bounding_boxes = []

        # Store the bounding boxes for collision detection
        collision_boxes = []

        # Place objects on the background image
        for _ in range(num_objects):
            # Randomly select an object image
            object_file = random.choice(object_png_files)

            # Get the class ID for the selected object
            class_id = object_class_ids[object_file]

            # Load your object image
            object_img = load_image(os.path.join(object_path, object_file))  # -1 to include the alpha channel if exists
            object_transformed = transform(image=object_img)['image']
            start_x = random.randint(0,abs( background_img.shape[1] - object_transformed.shape[1]))
            start_y = random.randint(0, abs(background_img.shape[0] - object_transformed.shape[0]))
            counter = 0
            # Check for collision and update position if needed
            # while is_collision([start_x, start_y, start_x + object_transformed.shape[1], start_y + object_transformed.shape[0]], collision_boxes) :
            #     start_x = random.randint(0, background_img.shape[1] - object_transformed.shape[1])
            #     start_y = random.randint(0, background_img.shape[0] - object_transformed.shape[0])
            collision_boxes.append([start_x, start_y, start_x + object_transformed.shape[1], start_y + object_transformed.shape[0]])
            # Save your bounding box coordinates in YOLO format
            bb_width = object_transformed.shape[1] / background_img.shape[1]
            bb_height = object_transformed.shape[0] / background_img.shape[0]
            x_center = (start_x + bb_width * background_img.shape[1] / 2) / background_img.shape[1]
            y_center = (start_y + bb_height * background_img.shape[0] / 2) / background_img.shape[0]

            bounding_boxes.append(f"{class_id} {x_center} {y_center} {bb_width} {bb_height}\n")

            # Overlay your object on the background image
            background_img = overlay_images(background_img, object_transformed, start_x, start_y)

        # Save your new image
        cv2.imwrite(os.path.join(output_dir, f'dataset{i}.{output_format}'), background_img)
        # Save your bounding box coordinates
        with open(os.path.join(output_dir, f'dataset{i}.txt'), 'w') as f:
            f.writelines(bounding_boxes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--youtube_video", help = "Get the youtube video", required= False)
    parser.add_argument("--fps", help = "how many frame per seconds in the dataset", required= False, default= 24)
    parser.add_argument("--video_path", help = "output directory of the dataset", required= False)
    parser.add_argument("--output_dir", help = "output directory of the dataset", required= True)
    parser.add_argument("--output_format", help = "output format of the dataset", required= True)
    parser.add_argument("--objects_dir", help = "directory of the objects", required= True )
    args = parser.parse_args()

    if args.video_path and args.youtube_video:
        print("You can't use both video_path and youtube_video")
        exit()
    elif args.youtube_video:
        getVideoFramesFromYoutube(args.youtube_video, args.output_dir, args.output_format, args.fps)
    elif args.video_path:
        getVideoFrames(args.video_path, args.output_dir, args.output_format, args.fps)

    # Create dataset with the embedded objects
    createDataset(args.output_dir, args.output_format, args.objects_dir)
