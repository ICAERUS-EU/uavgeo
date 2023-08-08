import os
try:
    import rioxarray as rx
except ImportError:
    rx = None

if rx ==None:
    raise ModuleNotFoundError(
                "Package `rioxarray`` is required to be installed to use uavgeo "
                "Please use `pip install rioxarary` or "
                "`conda install -c conda-forge rioxarray` "
                "to install the package"
                )
import yaml
from rioxarray import merge

def print_files(startpath):
    """
    Recursively prints all files and directories starting from a given path.

    Args:
        startpath (str): The path to start the recursive search from.

    Examples:
        >>> print_files("path/to/directory")
    """
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

def load_yolo_folder(yolo_path, subset="", img_ext = ".png", lab_ext = ".txt"):
    """
    Loads the file paths of YOLO labels and corresponding images from a specified folder.

    Args:
        yolo_path (str): The path to the YOLO folder.
        subset (str, optional): The subset folder to load (e.g., "train", "test", "val"). Default is "".
        img_ext (str, optional): The file extension of the images. Default is ".png".
        lab_ext (str, optional): The file extension of the labels. Default is ".txt".

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the list of label file paths and image file paths.

    Examples:
        >>> labels, images = load_yolo_folder("path/to/yolo", subset="train")
        >>> print(labels, images)
    """

    if subset == "":
        print("No train/test/val folder defined, is this your next step in the process?")

    folder_imgs = os.path.join(yolo_path, subset, "images")
    folder_labs = os.path.join(yolo_path, subset, "labels")

    imgs = [os.path.join(folder_imgs ,img) for img in os.listdir(folder_imgs) if img.endswith(img_ext)]
    labs = [os.path.join(folder_labs ,lab) for lab in os.listdir(folder_labs) if lab.endswith(lab_ext)]

    return (labs, imgs)

def check_labels_to_imgs(labs, imgs, img_fmt = ".png"):
    """
    Checks if there is a matching label file for each image file based on their file paths.

    Args:
        labs (List[str]): The list of label file paths.
        imgs (List[str]): The list of image file paths.
        img_fmt (str, optional): The expected image file format. Default is ".png".

    Examples:
        >>> labels = ["path/to/labels/1.txt", "path/to/labels/2.txt"]
        >>> images = ["path/to/images/1.png", "path/to/images/2.png"]
        >>> check_labels_to_imgs(labels, images)
    """
    l_labs = len(labs)
    l_imgs = len(imgs)
    if l_labs != l_imgs:
        print("Found difference in the two objects. Images: {}, Labels {}".format(l_imgs, l_labs))
    for lab, img in zip(labs,imgs):
        expected_img = lab.replace("labels", "images")
        expected_img = expected_img.replace(".txt", ".png")
        if expected_img != img:
            print("Found difference between image/label file: expected: {}, found: {}".format(expected_img, img))

def generate_yolo_yaml(local_working_dir, root_folder, output_yaml, class_dict):
    """
    Generates a YAML file for YOLO format based on the provided inputs.

    Args:
        local_working_dir (str): The local working directory path.
        root_folder (str): The root folder path.
        output_yaml (str): The output YAML file path to be generated.
        class_dict (dict): A dictionary mapping class indices to class names.

    Returns:
        None

    Raises:
        IOError: If there is an error opening or writing to the output YAML file.

    Example:
        generate_yolo_yaml('path/to/local_working_dir', 'path/to/root_folder',
                           'output.yaml', {0: 'class1', 1: 'class2', 2: 'class3'})
    """
    # Open the output YAML file for writing
    data = {
        "path": local_working_dir,
        "train": os.path.join(root_folder, "train", "images"),
        "val": os.path.join(root_folder, "val", "images"),
        "test": os.path.join(root_folder, "test", "images"),
        "names": class_dict,
    }
    with open(output_yaml, 'w') as file:
        yaml.dump(data, file)
    return data

