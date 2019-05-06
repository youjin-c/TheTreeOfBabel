#open json files and add node list from log.txt files
import glob
import sys
import re
import os
from pathlib import Path
import json
from collections import Counter
import subprocess

def refine(path1,path2):
    for entry in sorted(os.scandir(path1), key=lambda x: (x.is_dir(), x.name)):
        filenameText = entry.name.split('.')[0]
        if filenameText.isdigit():
            # print(["cp",path1+entry.name,path2+"basic"])
            subprocess.run(["cp",path2+entry.name,path2+"cycle"])


refine(sys.argv[1],sys.argv[2])
