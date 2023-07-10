import shutil
import os
from urllib import request
from tqdm import tqdm

# straight from the tqdm docs:

def my_hook(t):
  """
  Wraps tqdm instance. Don't forget to close() or __exit__()
  the tqdm instance once you're done with it (easiest using `with` syntax).

  Example
  -------

  >>> with tqdm(...) as t:
  ...     reporthook = my_hook(t)
  ...     urllib.urlretrieve(..., reporthook=reporthook)

  """
  last_b = [0]

  def inner(b=1, bsize=1, tsize=None):
    """
    b  : int, optional
        Number of blocks just transferred [default: 1].
    bsize  : int, optional
        Size of each block (in tqdm units) [default: 1].
    tsize  : int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
    """
    if tsize is not None:
        t.total = tsize
    t.update((b - last_b[0]) * bsize)
    last_b[0] = b
  return inner

   
def download(url, filename = None,redownload =False,output_dir ="data", type= "" ):

    """

    Wraps a few libraries together to download (and extract if required) a file from https.
    It outputs the filepath it has stored to.

    url     : str 
        url of the file: can be anything in theory, as long as it links directly to the file
    output_dir   : str
        preferred output path, e.g. "data" [default = "data"]
    type    : str 
        subfolder to create based on datatype: e.g."raw", gets appended to output_dir [default = ""]
    filename : str, optional 
        force a filename to store into , please use the correct file-output type, e.g. .zip/.tif./.tar.bz2 [default = None]
    redownload   : str, optional
        force to redownload if the file already exists [default = False].
    
    Returns: output_path of file 
    
    Example
  -------

  >>> fname = download("https://fastly.picsum.photos/id/237/200/300.jpg", filename = "test.jpg", type = "raw")

  """


    if filename is None:
        filename = url.split("/")[-1]
    
    path = os.path.join(output_dir, type)

    if not os.path.exists(path):
        os.mkdir(path)

    fpath = os.path.join(path, filename)
    
    if os.path.exists(fpath) and redownload==False:
        print("Already downloaded file, please add redownload=True to redownload")
        return fpath
    
    with tqdm(unit='B', unit_scale=True, leave=True, miniters=1, desc=url.split('/')[-1]) as t: 
        # all optional kwargs
              request.urlretrieve(url, fpath, reporthook=my_hook(t), data=None)
    
    
    if fpath.endswith(".zip") or fpath.endswith(".rar") or fpath.endswith(".tar.bz2") :
        
        shutil.unpack_archive(fpath,path)
        
    return fpath