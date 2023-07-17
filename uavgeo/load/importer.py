import os
try:
    import rioxarray as rx
except ImportError:
    rx = None
import torchdata

if rx ==None:
    raise ModuleNotFoundError(
                "Package `rioxarray`` is required to be installed to use uavgeo "
                "Please use `pip install rioxarary` or "
                "`conda install -c conda-forge rioxarray` "
                "to install the package"
                )


def load_sfm(path_to_file = None, xr_name = "rgb_ortho"):
    """
    Loads an image from a specified file path using the rioxarray library.

    Args:
        path_to_file (str, optional): The file path of the satellite image. Default is "data/rgb/sfm/ortho.tif".
        xr_name (str, optional): The name to assign to the xarray DataArray. Default is "rgb_ortho".

    Returns:
        xarray.DataArray: The loaded image.

    Examples:
        >>> img = load_sfm())
        >>> print(img)
    """
    if path_to_file is None:
        path_to_file = "data/rgb/sfm/ortho.tif"
    f = rx.open_rasterio(path_to_file, default_name = xr_name)
    return f

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
