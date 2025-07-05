import os
from PIL import Image

def png_to_gif(input_folder, output_gif, duration=200, loop=0):
    """
    Convert all PNG files in a folder to an animated GIF.

    Parameters:
        input_folder (str): Path to folder containing PNG files
        output_gif (str): Path to save the output GIF
        duration (int): Duration between frames in milliseconds
        loop (int): Number of loops (0 for infinite)
    """
    # Get all PNG files from the folder and sort them
    png_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]
    png_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))  # Sort to ensure correct order

    if not png_files:
        print("No PNG files found in the specified folder.")
        return

    # Open all images and append to a list
    images = []
    for png_file in png_files:
        file_path = os.path.join(input_folder, png_file)
        try:
            img = Image.open(file_path)
            images.append(img)
        except Exception as e:
            print(f"Could not open {png_file}: {e}")

    if not images:
        print("No valid PNG files could be opened.")
        return

    # Save as GIF
    try:
        images[0].save(
            output_gif,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop
        )
        print(f"Successfully created GIF at {output_gif}")
    except Exception as e:
        print(f"Error creating GIF: {e}")


# Example usage
if __name__ == "__main__":
    input_folder = "./plots"  # Change this to your folder path
    output_gif = "output.gif"  # Output GIF filename

    png_to_gif(input_folder, output_gif, duration=50, loop=0)