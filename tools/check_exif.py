import argparse
import os

from PIL import Image


def check_exif_orientation(image_path):
    """
    Check the EXIF orientation of an image.
    Returns:
        int: Orientation value (1-8), or None if no EXIF/Orientation found.
    """
    try:
        with Image.open(image_path) as img:
            # 274 is the tag ID for 'Orientation'
            exif = img._getexif()
            if exif is not None:
                return exif.get(274)
    except Exception:
        # Skip files that are corrupted or cannot be opened
        return None
    return None


def scan_recursive(input_root):
    """
    Recursively scan all subdirectories for JPG images with non-normal EXIF.
    """
    valid_extensions = (".jpg", ".jpeg", ".JPG", ".JPEG")
    risky_files = []
    total_count = 0

    print(f"Scanning recursively starting from: {input_root}")

    # os.walk(root) recursively yields (current_path, sub_dirs, files)
    for root, dirs, files in os.walk(input_root):
        for filename in files:
            if filename.lower().endswith(valid_extensions):
                total_count += 1
                full_path = os.path.join(root, filename)

                orientation = check_exif_orientation(full_path)

                # If orientation exists and is NOT 1 (Normal), it's a risky file for DALI
                if orientation is not None and orientation != 1:
                    # Calculate the path relative to the input_root
                    relative_path = os.path.relpath(full_path, input_root)
                    risky_files.append((relative_path, orientation))

    # --- Results Metadata ---
    print("\n" + "=" * 60)
    print("Scan Summary:")
    print(f"Total JPG images analyzed: {total_count}")
    print(f"Images with NON-NORMAL EXIF: {len(risky_files)}")
    print("=" * 60 + "\n")

    if risky_files:
        print(f"{'RELATIVE PATH':<50} | {'EXIF TAG'}")
        print("-" * 65)
        for rel_path, tag in risky_files:
            print(f"{rel_path:<50} | {tag}")
    else:
        print("Great! All images are standard (Orientation 1 or no EXIF).")


def main():
    parser = argparse.ArgumentParser(description="Recursively scan subfolders for images with EXIF Orientation != 1.")
    parser.add_argument("--input", type=str, required=True, help="Root directory to start recursive scan.")

    args = parser.parse_args()

    # Convert to absolute path to ensure correct relative path calculation
    abs_input_path = os.path.abspath(args.input)

    if not os.path.isdir(abs_input_path):
        print(f"Error: Target path '{args.input}' is not a directory.")
        return

    scan_recursive(abs_input_path)


if __name__ == "__main__":
    main()
