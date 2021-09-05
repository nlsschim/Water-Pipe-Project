import sys
import os
from setuptools import setup, find_packages
PACKAGES = find_packages()

# Get version and release info, which is all stored in Water-Pipe-Project/version.py
ver_file = os.path.join('water_main_predictions', 'version.py')
with open(ver_file) as f:
    exec(f.read())

# Give setuptools a hint to complain if it's too old a version
# 24.2.0 added the python_requires option
# Should match pyproject.toml
# SETUP_REQUIRES = ['setuptools >= 24.2.0']
# This enables setuptools to install wheel on-the-fly
# SETUP_REQUIRES += ['wheel'] if 'bdist_wheel' in sys.argv else []
PACKAGES = find_packages()

opts = dict(name='water_pipe_predictions',
            maintainer='Nels Schimek',
            maintainer_email='nlsschim@uw.edu',
            description='Package for predicting pipe breaks',
            long_description='Package to use a Random Forest Machine learning model to predict pipe breaks and determine most important features',
            url='https://github.com/nlsschim/Water-Pipe-Project',
            download_url='https://github.com/nlsschim/Water-Pipe-Project',
            license='MIT',
            classifiers=CLASSIFIERS,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            version='0.1.0',
            packages=PACKAGES,
            package_data=PACKAGE_DATA,
            python_requires=PYTHON_REQUIRES,
            #setup_requires=SETUP_REQUIRES,
            install_requires=REQUIRES)


if __name__ == '__main__':
    setup(**opts)
