import os
import zipfile

zf = zipfile.ZipFile("houseganpp_pyzip.zip", "w")
for dirname, subdirs, files in os.walk("/home/evalexii/Documents/IAAIP/Repos/houseganpp"):
    zf.write(dirname)
    for filename in files:
        zf.write(os.path.join(dirname, filename))
zf.close()
