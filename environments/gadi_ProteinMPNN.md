
# Step 0:
Make sure it is not already installed!!! **It should already be on Gadi**


# Step 1: Create env

```
#navigate to correct dir
cd /path/to/install/location

# load modules
module load  python3/3.11.7 cuda/12.9.0

# Make env
python3 -m venv pmpnn

# Load env
source pmpnn/bin/activate

#install dependencies
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

#clone ProteinMPNN
cd /path/to/repo/dir
git clone https://github.com/dauparas/ProteinMPNN.git
```

# Step 2: Clean env
```
# change group to xz2 and give read write privilages
chgrp xz2 ./pmpnn
chmod g+ws ./pmpnn

# check inode usage before cleaning
find ./pmpnn -xdev -printf '.' | wc -c

# Safe deployment cleanup: remove generated caches only.
# Do not delete package directories such as "tests" or "doc*"; some packages
# import from those directories at runtime.
find ./pmpnn -type d -name "__pycache__" -exec rm -rf {} +
find ./pmpnn -name "*.pyc" -delete
find ./pmpnn -name "*.pyo" -delete
find ./pmpnn -name ".DS_Store" -delete
find ./pmpnn -type d -name ".pytest_cache" -exec rm -rf {} +

# check inode usage after cleaning
find ./pmpnn -xdev -printf '.' | wc -c
```
# Step 3: Loading Dir
```
module load  python3/3.11.7 cuda/12.9.0
source /path/to/install/location/pmpnn/bin/activate
```
