
# Step 0:
Make sure it is not already installed!!! **It should already be on Gadi**


# Step 1: Create env

```
# load modules
module load cuda/12.9.0 python3/3.10.4

# create venv
python3 -m venv colabdesign_modified

# Load env
source colabdesign_modified/bin/activate

#install dependencies
pip install -r gadi_colabdesign_requirments.txt
```

# Step 2: Clean env
```
# change group to xz2 and give read write privilages
chgrp xz2 ./colabdesign_modified
chmod g+ws ./colabdesign_modified

# check inode usage before cleaning
find ./colabdesign_modified -xdev -printf '.' | wc -c

# Remove documentation directories
find ./colabdesign_modified -type d -name "doc*" -exec rm -rf {} +

# Remove test directories  
find ./colabdesign_modified -type d -name "tests" -exec rm -rf {} +

# Remove Python cache files
find ./colabdesign_modified -type d -name "__pycache__" -exec rm -rf {} +

# Remove additional cache and temp files
find ./colabdesign_modified -name "*.pyc" -delete
find ./colabdesign_modified -name "*.pyo" -delete
find ./colabdesign_modified -name ".DS_Store" -delete
find ./colabdesign_modified -type d -name ".pytest_cache" -exec rm -rf {} +

# check inode usage after cleaning
find ./colabdesign_modified -xdev -printf '.' | wc -c
```
# Step 3: Loading Dir
```
module load cuda/12.9.0 python3/3.10.4
source /path/to/install/location/colabdesign_modified/bin/activate
```
