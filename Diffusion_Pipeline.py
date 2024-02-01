import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import load_image, make_image_grid
import torch
import argparse


# Function to create a square mask
def create_square_mask(image_size, square1_coords, square2_coords):
    square_mask = torch.zeros(image_size)
    square_mask[square1_coords[0]:square1_coords[2]+1, square1_coords[1]:square1_coords[3]+1] = 1
    square_mask[square2_coords[0]:square2_coords[2]+1, square2_coords[1]:square2_coords[3]+1] = 1
    return square_mask
def mask_size():
  # Image size (adjust as needed)
  image_size = (256, 256)

  # Square coordinates (min_y, min_x, max_y, max_x)
  square1_coords = (50, 10, 150, 110)
  square2_coords = (100, 120, 200, 220)

  # Create the square mask
  square_mask = create_square_mask(image_size, square1_coords, square2_coords)

  # Convert the mask to a PyTorch tensor
  tensor_mask = torch.unsqueeze(square_mask, 0).unsqueeze(0)

  # Visualize and save the square mask
  plt.title('Square Mask')
  plt.savefig('/content/square_mask.png')

# Inference
# this is the inference code that you can use after you have trained your model
# Unhide code below and change prj_path to your repo or local path (e.g. my_dreambooth_project)
#
#
#
def inference(model_path: str, img_path: str, masking: str, prompt_text: str, num_steps: str, num_images: int):
  prj_path = model_path
  img_url = img_path
  mask_url = masking
  model = "stabilityai/stable-diffusion-xl-base-1.0"
  init_image = load_image(img_url).resize((512, 512))
  mask_image = load_image(mask_url).resize((512, 512))
  repo_id = "stabilityai/stable-diffusion-2-inpainting"
  pipe = DiffusionPipeline.from_pretrained(
      model,
      torch_dtype=torch.float16,
  )
  pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
  pipe.to("cuda")
  pipe.load_lora_weights(prj_path, weight_name="pytorch_lora_weights.safetensors")
  # refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
  #     "stabilityai/stable-diffusion-xl-refiner-1.0",
  #     torch_dtype=torch.float16,
  # )
  # refiner.to("cuda")
  prompt = prompt_text
  seed = 42
  generator = torch.Generator("cuda").manual_seed(seed)
  num_images = num_images
  generated_images = []
  for i in range(num_images):
      image = pipe(prompt=prompt, image=init_image, mask_image=mask_image, num_inference_steps=num_steps).images[0]
      
      
      # Save the generated image
      image_path = f"generated_image_{i+1}.png"
      image.save(image_path)
      generated_images.append(image_path)

  # Display the generated images in a grid
  all_images = [init_image, mask_image] + generated_images
  make_image_grid(all_images, rows=1, cols=num_images + 2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help = "path to the model", required= True)
    parser.add_argument("--img_path", help = "path to the initial image", required= True)
    parser.add_argument("--masking", help = "path to the image after masking", required= True)
    parser.add_argument("--prompt_text", help = "prompt to generate the image", required= True)
    parser.add_argument("--num_steps", help = "numbers of inference steps", required= True)
    parser.add_argument("--num_images", help = "numbers of images to generate", required= True)
    args = parser.parse_args()

    mask_size()
    inference(args.prj_path, args.img_url, args.masking, args.prompt, args.steps, args.num_images)
