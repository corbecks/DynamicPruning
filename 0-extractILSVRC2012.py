import os
import tarfile


def extract_tar_gz_files(src_root, dest_root):
    # Walk through the source root directory
    for dirpath, _, filenames in os.walk(src_root):
        for filename in filenames:
            if filename.endswith('.tar.gz'):
                # Construct the full path of the .tar.gz file
                tar_gz_path = os.path.join(dirpath, filename)

                # Create the destination folder path by replacing the source root with the destination root
                relative_path = os.path.relpath(dirpath, src_root)
                dest_dir = os.path.join(dest_root, relative_path, os.path.splitext(os.path.splitext(filename)[0])[0])

                # Make sure the destination folder exists, create if it doesn't
                os.makedirs(dest_dir, exist_ok=True)

                # Extract the tar.gz file into the destination folder
                with tarfile.open(tar_gz_path, 'r:gz') as tar:
                    tar.extractall(path=dest_dir)

                print(f"Extracted {tar_gz_path} to {dest_dir}")


# Example usage
src_root = "C:\\Users\\cmbec\\OneDrive\\Cloud_Documents\\Harvard\\NEUROBIO240\\AlexNet\\ILSVRC2012_img_train"  # Path to the root folder with tar.gz files
dest_root = "C:\\Users\\cmbec\\OneDrive\\Cloud_Documents\\Harvard\\NEUROBIO240\\AlexNet\\ILSVRC2012_img_train_extracted"  # Path to the new root directory

extract_tar_gz_files(src_root, dest_root)