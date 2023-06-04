import pytube 
import os
import subprocess
import cv2
import numpy as np
import random
import argparse
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

import os
import cv2
import random
import numpy as np

def is_collision(new_box, existing_boxes):
    for existing_box in existing_boxes:
        if (new_box[0] < existing_box[2] and
            new_box[2] > existing_box[0] and
            new_box[1] < existing_box[3] and
            new_box[3] > existing_box[1]):
            return True
    return False

def matrix_from_euler_xyz(euler_angles):
    # Assuming the angles are in radians.
    roll = euler_angles[0]
    pitch = euler_angles[1]
    yaw = euler_angles[2]

    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    R_x = np.array([[1, 0, 0],
                    [0, cos_roll, -sin_roll],
                    [0, sin_roll, cos_roll]])

    R_y = np.array([[cos_pitch, 0, sin_pitch],
                    [0, 1, 0],
                    [-sin_pitch, 0, cos_pitch]])

    R_z = np.array([[cos_yaw, -sin_yaw, 0],
                    [sin_yaw, cos_yaw, 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

def createDataset(output_dir: str, output_format: str, object_path: str):
    # Ensure dataset directory exists, if not, create it
    os.makedirs(output_dir, exist_ok=True)

    # List all files in directory
    files = os.listdir(output_dir)
    object_files = os.listdir(object_path)

    # Filter out all non-png files
    png_files = [i for i in files if i.endswith(f'.{output_format}')]
    object_png_files = [i for i in object_files if i.endswith(f'.{output_format}')]

    # Assign each object a unique class ID
    object_class_ids = {object_file: i for i, object_file in enumerate(object_png_files)}

    # Sort files alphabetically to maintain a consistent order
    png_files.sort()
    object_png_files.sort()

    # Loop over your background images
    for i, file in enumerate(png_files):
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
            object_img = cv2.imread(os.path.join(object_path, object_file), -1)  # -1 to include the alpha channel if exists
            
            # Create a random scale factor
            scale = min(random.uniform(0.2, 1.5), min(background_img.shape[0]/object_img.shape[0], background_img.shape[1]/object_img.shape[1]))
            
            # Resize the object image
            object_img_resized = cv2.resize(object_img, None, fx=scale, fy=scale)
            # Create a random rotation angle
            angle = random.uniform(-180, 180)
            
            # Get the rotation matrix for 2D rotation
            (h, w) = object_img_resized.shape[:2]
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Rotate the object image for 2D rotation
            object_img_rotated = cv2.warpAffine(object_img_resized, M, (w, h))

            # Create a random pitch, yaw, roll for 3D rotation
            pitch = random.uniform(-np.pi, np.pi)
            yaw = random.uniform(-np.pi, np.pi)
            roll = random.uniform(-np.pi, np.pi)

            # Create the rotation matrix for 3D rotation
            # R = matrix_from_euler_xyz([pitch, yaw, roll])

            # Apply the 3D rotation
            # object_img_rotated = cv2.warpPerspective(object_img_rotated, R[:2, :2], (object_img_rotated.shape[1], object_img_rotated.shape[0]))
            # Rotate the object image for 2D rotation
            object_img_rotated = cv2.warpAffine(object_img_resized, M, (w, h))
            # Select a random location to place the object
            start_x = random.randint(0, background_img.shape[1] - object_img_rotated.shape[1])
            start_y = random.randint(0, background_img.shape[0] - object_img_rotated.shape[0])

            # Check for collision and update position if needed
            while is_collision([start_x, start_y, start_x + object_img_rotated.shape[1], start_y + object_img_rotated.shape[0]], collision_boxes):
                start_x = random.randint(0, background_img.shape[1] - object_img_rotated.shape[1])
                start_y = random.randint(0, background_img.shape[0] - object_img_rotated.shape[0])

            collision_boxes.append([start_x, start_y, start_x + object_img_rotated.shape[1], start_y + object_img_rotated.shape[0]])

            # Split the object image into BGR channels and Alpha channel
            b, g, r, alpha = cv2.split(object_img_rotated)

            # Create a 3 channel image and a mask from alpha channel for blending
            object_rgb = cv2.merge([b, g, r])
            alpha = cv2.merge([alpha, alpha, alpha]) / 255.0

            # Position the object onto the background using alpha mask for blending
            background_img[start_y:start_y+object_img_rotated.shape[0], start_x:start_x+object_img_rotated.shape[1]] = \
                alpha * object_rgb + (1 - alpha) * background_img[start_y:start_y+object_img_rotated.shape[0], start_x:start_x+object_img_rotated.shape[1]]

            # Save your bounding box coordinates in YOLO format
            bb_width = object_img_rotated.shape[1] / background_img.shape[1]
            bb_height = object_img_rotated.shape[0] / background_img.shape[0]
            x_center = (start_x + bb_width * background_img.shape[1] / 2) / background_img.shape[1]
            y_center = (start_y + bb_height * background_img.shape[0] / 2) / background_img.shape[0]

            bounding_boxes.append(f"{class_id} {x_center} {y_center} {bb_width} {bb_height}\n")

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