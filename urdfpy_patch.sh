#!/bin/bash

# This script patches networkx for Urdfpy such that Urdfpy can work with
# python 3.10. This script assumes that a conda environment is active
# and using python 3.10

# Find python directory
# If command line specified had --docker arg, then this assumes an official isaac lab or
# isaac sim docker is in use. For these, isaac sim is installed at the root.

arg=
if [ $# -eq 0 ]; then
    arg="none"
else
  if [ "$1" = "--docker" ]; then
    arg="--docker"
  else
    echo "Incorrect command line parameter. Only --docker is a valid command line param."
    exit 1
  fi
fi

python_dir=
if [ "$arg" = "--docker" ]; then
  python_dir=$(/isaac-sim/python.sh -m pip show "networkx" | grep "Location:" | awk '{print $2}')
else
  python_dir=$(pip show "networkx" | grep "Location:" | awk '{print $2}')
fi

# Base directory of networkx
base_dir="${python_dir}/networkx"

# Patching graph.py ------------------------------
file="$base_dir/classes/graph.py"
old_code="from collections import Mapping"
new_code="from collections.abc import Mapping"

# Check if the file exists
if [ ! -f "$file" ]; then
    echo "File not found: $file"
    exit 1
fi

# Use sed to replace code in file
sed -i "s|$old_code|$new_code|g" "$file"

# Patching coreviews.py ------------------------------
file="$base_dir/classes/coreviews.py"
old_code="from collections import Mapping"
new_code="from collections.abc import Mapping"

# Check if the file exists
if [ ! -f "$file" ]; then
    echo "File not found: $file"
    exit 1
fi

# Use sed to replace code in file
sed -i "s|$old_code|$new_code|g" "$file"

# Patching reportviews.py ------------------------------
file="$base_dir/classes/reportviews.py"
old_code="from collections import Mapping, Set, Iterable"
new_code="from collections.abc import Mapping, Set, Iterable"

# Check if the file exists
if [ ! -f "$file" ]; then
    echo "File not found: $file"
    exit 1
fi

# Use sed to replace code in file
sed -i "s|$old_code|$new_code|g" "$file"

# Patching dag.py ------------------------------
file="$base_dir/algorithms/dag.py"
old_code="from fractions import gcd"
new_code="from math import gcd"

# Check if the file exists
if [ ! -f "$file" ]; then
    echo "File not found: $file"
    exit 1
fi

# Use sed to replace code in file
sed -i "s|$old_code|$new_code|g" "$file"


# Patching lowest_common_ancestors.py ------------------------------
file="$base_dir/algorithms/lowest_common_ancestors.py"
old_code="from collections import defaultdict, Mapping, Set"
new_code1="from collections.abc import Mapping, Set"
new_code2="from collections import defaultdict"

# Check if the file exists
if [ ! -f "$file" ]; then
    echo "File not found: $file"
    exit 1
fi

# Use sed to replace code in file
sed -i "/$old_code/c\\
$new_code1\\
$new_code2" "$file"

# Patching graphml.py so that it uses int instead of np.int (which has been deprecated)
file="$base_dir/readwrite/graphml.py"
old_code='(np.int, "int"), (np.int8, "int"),'
new_code='(int, "int"), (np.int8, "int"),'

# Check if the file exists
if [ ! -f "$file" ]; then
    echo "File not found: $file"
    exit 1
fi

# Use sed to replace code in file
sed -i "s|$old_code|$new_code|g" "$file"

# Patching urdfpy so that it uses float instead of np.float (which has been deprecated)
# Base directory of urdfpy
base_dir="${python_dir}/urdfpy"
file="$base_dir/urdf.py"
old_code="value = np.asanyarray(value).astype(np.float)"
new_code="value = np.asanyarray(value).astype(float)"

# Check if the file exists
if [ ! -f "$file" ]; then
    echo "File not found: $file"
    exit 1
fi

# Use sed to replace code in file
sed -i "s|$old_code|$new_code|g" "$file"
