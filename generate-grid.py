import matplotlib.pyplot as plt
import cv2
import os

def create_grid():
    # CHANGE THIS to the exact name of one of the images in your data folder (without the .png/.jpg extension)
    image_name = "building_5aca8b_google"
    
    orig_path = f"data/{image_name}.png"
    if not os.path.exists(orig_path):
        orig_path = f"data/{image_name}.jpg" # fallback if it's a jpg
        
    classical_path = f"out/{image_name}_classical.jpg"
    sr_path = f"out/{image_name}_sr.jpg"

    if not all(os.path.exists(p) for p in [orig_path, classical_path, sr_path]):
        print("Error: Run enhance.py on your data first to generate the classical and sr outputs.")
        return

    # Load images
    orig = cv2.imread(orig_path)
    classical = cv2.imread(classical_path)
    sr = cv2.imread(sr_path)

    # Convert BGR to RGB
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    classical = cv2.cvtColor(classical, cv2.COLOR_BGR2RGB)
    sr = cv2.cvtColor(sr, cv2.COLOR_BGR2RGB)

    # Create the grid
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(orig)
    axes[0].set_title('Original (640x640)')
    axes[0].axis('off')

    axes[1].imshow(classical)
    axes[1].set_title('Task 1: Classical Enhancement')
    axes[1].axis('off')

    axes[2].imshow(sr)
    axes[2].set_title('Task 2: FSRCNN Super-Resolution')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('visual_grid.jpg', dpi=150)
    print("Saved visual_grid.jpg to your project folder.")

if __name__ == "__main__":
    create_grid()