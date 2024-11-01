#!/usr/bin/env python3

## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# Get values from package.xml:
setup_args = generate_distutils_setup(
    packages=['tum_tb_perception'],
    package_dir={'': 'src'},
)

setup(**setup_args)
