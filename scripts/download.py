import collections
import logging
import os
import re
import sys
import tarfile
import urllib.request
import zipfile
from typing import Any, List, Optional, overload, Union, Dict, Tuple

import numpy as np

path = './style_classifiers'
filepath = './style_classifiers/yelp-bert-base-uncased-train.tar.gz'

logging.info("Extract %s", filepath)
if tarfile.is_tarfile(filepath):
    with tarfile.open(filepath, "r") as tfile:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tfile, path)
elif zipfile.is_zipfile(filepath):
    with zipfile.ZipFile(filepath) as zfile:
        zfile.extractall(path)
else:
    logging.info(
        "Unknown compression type. Only .tar.gz"
        ".tar.bz2, .tar, and .zip are supported"
    )