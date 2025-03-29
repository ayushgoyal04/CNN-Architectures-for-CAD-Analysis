import os
import glob

def rename_images(folder_path, start_num):
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    images = sorted(glob.glob(os.path.join(folder_path, '*')))  # Get all images

    if len(images) == 0:
        print("No images found in the folder.")
        return

    for i, img_path in enumerate(images):
        ext = os.path.splitext(img_path)[1]  # Get file extension
        new_name = f"{start_num + i}{ext}"
        new_path = os.path.join(folder_path, new_name)

        try:
            os.rename(img_path, new_path)
        except Exception as e:
            print(f"Error renaming {img_path}: {e}")

    print("Renaming complete!")

# Example usage
folder = r"D:\image processing group\processed_images_main\train\1"  # Use raw string (r"")
m = 1  # Starting number
rename_images(folder, m)
