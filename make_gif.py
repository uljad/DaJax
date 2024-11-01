import os
import glob
from PIL import Image, ImageDraw, ImageFont
import re

def extract_number(filename):
    # Extract the last number before .png
    number = filename.split("/")[-1].split("_")[-1].split(".")[0]
    return int(number)
    
def create_gif(image_dir='data/media/gif', output_path='animation.gif', duration=200):
    """
    Create a GIF from PNG images in the specified directory.
    """
    # Get list of PNG files
    png_files = glob.glob(os.path.join(image_dir, '*.png'))
    png_files.sort(key=extract_number)
    
    print(f"Found {len(png_files)} PNG files")
    
    if not png_files:
        print(f"No PNG files found in {image_dir}")
        return
    
    # Get base image size from first image
    base_img = Image.open(png_files[0])
    img_w, img_h = base_img.size
    
    # Add padding for text
    padding = 50
    new_h = img_h + padding
    
    # Create list of images
    frames = []
    for png_file in png_files:
        try:
            # Open and process each image
            img = Image.open(png_file)
            
            # Create new image with space for text
            new_img = Image.new('RGB', (img_w, new_h), 'white')
            new_img.paste(img, (0, padding))
            
            # Add text
            draw = ImageDraw.Draw(new_img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
            except:
                font = ImageFont.load_default()
            
            # Get checkpoint number
            checkpoint_num = extract_number(png_file)
            text = f"Checkpoint {checkpoint_num}"
            
            # Center the text
            text_width = draw.textlength(text, font=font)
            x = (img_w - text_width) // 2
            draw.text((x, 10), text, fill='black', font=font)
            
            frames.append(new_img)
            print(f"Processed: {png_file}")
            
        except Exception as e:
            print(f"Error processing {png_file}: {str(e)}")
    
    if not frames:
        print("No frames were created")
        return
    
    # Save the GIF
    try:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1::10],
            duration=duration,
            loop=0
        )
        print(f"GIF created successfully at {output_path}")
    except Exception as e:
        print(f"Error saving GIF: {str(e)}")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('data/media', exist_ok=True)
    
    create_gif(
        image_dir='data/media/gif',
        output_path='animation_new_rare.gif',
        duration=200  # Half a second per frame
    )
