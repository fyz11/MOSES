#!/usr/bin/env python

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(name='MOSES',
      version='1.0',
      description='Python library for Long-time Biological Motion Analysis',
      author='Felix Y. Zhou',
      license='Ludwig Non-Commercial License',
      author_email='felixzhou1@gmail.com',
      zip_safe=False,
      install_requires=[
        "numpy",
        "pillow",
        "opencv-python>=3.0,<4.0",
        "matplotlib",
        "scipy",
        "tqdm",
        "tifffile",
        "scikit-image",
        "scikit-learn",
      ],
      packages=find_packages(exclude=('tests', 'docs'))
)