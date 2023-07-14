import os
import rioxarray as rx
import torchdata
import image_bbox_slicer


def load_sfm(path_to_file = None, xr_name = "rgb_ortho"):
    if path_to_file is None:
        path_to_file = "data/rgb/sfm/ortho.tif"
    f = rx.open_rasterio(path_to_file, default_name = xr_name)
    return f

def print_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

def load_yolo_folder(yolo_path, subset="", img_ext = ".png", lab_ext = ".txt"):

    if subset == "":
        print("No train/test/val folder defined, is this your next step in the process?")

    folder_imgs = os.path.join(yolo_path, subset, "images")
    folder_labs = os.path.join(yolo_path, subset, "labels")

    imgs = [os.path.join(folder_imgs ,img) for img in os.listdir(folder_imgs) if img.endswith(img_ext)]
    labs = [os.path.join(folder_labs ,lab) for lab in os.listdir(folder_labs) if lab.endswith(lab_ext)]

    return (labs, imgs)

def check_labels_to_imgs(labs, imgs, img_fmt = ".png"):
    l_labs = len(labs)
    l_imgs = len(imgs)
    if l_labs != l_imgs:
        print("Found difference in the two objects. Images: {}, Labels {}".format(l_imgs, l_labs))
    for lab, img in zip(labs,imgs):
        expected_img = lab.replace("labels", "images")
        expected_img = expected_img.replace(".txt", ".png")
        if expected_img != img:
            print("Found difference between image/label file: expected: {}, found: {}".format(expected_img, img))
