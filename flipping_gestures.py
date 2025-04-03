import os
import cv2

def flip_gesture_images(src_folder, dest_folder, gesture_name):
    """
    Flip images from the source folder and store them in the destination folder with a new gesture name.
    :param src_folder: Path to the source folder (e.g., 'scroll_up')
    :param dest_folder: Path to the destination folder (e.g., 'scroll_down')
    :param gesture_name: Name of the gesture being flipped (e.g., 'scroll_up' to 'scroll_down')
    """
    os.makedirs(dest_folder, exist_ok=True)
    
    # Loop through all images in the source folder and flip them
    for i, filename in enumerate(os.listdir(src_folder)):
        if filename.endswith(".jpg"):  # Check if the file is a .jpg image
            try:
                # Load the image
                img_path = os.path.join(src_folder, filename)
                img = cv2.imread(img_path)

                if img is None:  # Check if the image is loaded correctly
                    print(f"Skipping invalid image: {filename}")
                    continue

                # Flip the image horizontally (left to right)
                flipped_img = cv2.flip(img, 0)

                # Define the new filename for the flipped image
                flipped_file_path = os.path.join(dest_folder, f"{gesture_name}{i}.jpg")

                # Save the flipped image
                cv2.imwrite(flipped_file_path, flipped_img)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"Skipping non-image file: {filename}")

    print(f"Flipping complete. {gesture_name} images saved to {dest_folder}")

if __name__ == "__main__":  # Corrected the condition
    src_scroll_up_folder = r"E:\Sign_language\Sign-language-recognition\demo\scroll_up"
    dest_scroll_down_folder = r"E:\Sign_language\Sign-language-recognition\demo\scroll_down"
    # Flip scroll_up images to create scroll_down images
    flip_gesture_images(src_scroll_up_folder, dest_scroll_down_folder, "scroll_down")
