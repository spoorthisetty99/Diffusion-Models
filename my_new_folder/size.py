import os
from PIL import Image

# Folder containing images
input_folder = "/home/user/DDPM/pytorch-ddpm-master/raw-890"
output_folder = "//home/user/DDPM/pytorch-ddpm-master/UIEB"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get list of PNG image files
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]
image_files.sort()  # Ensure sequential ordering

# Resize and rename images
for i, filename in enumerate(image_files, start=1):
    img_path = os.path.join(input_folder, filename)
    img = Image.open(img_path).convert("RGBA")  # Ensure PNG transparency is preserved
    img_resized = img.resize((256, 256), Image.LANCZOS)  # Resize with high-quality filter
    
    # Save with new sequential name
    new_filename = f"{i}.png"
    img_resized.save(os.path.join(output_folder, new_filename), "PNG")

print(f"Processed {len(image_files)} PNG images and saved them to {output_folder}")
