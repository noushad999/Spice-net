import urllib.request
import os

url  = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
dest = "/mnt/d/SpiceNet/mobile_sam.pt"

if os.path.exists(dest):
    print(f"Already exists: {dest}")
else:
    print("Downloading MobileSAM checkpoint (~38MB)...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to: {dest}")
