#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name="xmap_coordinates",
    version="0.0.1",
    description="",
    long_description="",
    author="",
    author_email="mcdevitts@gmail.com",
    url="https://github.com/mcdevitts/xmap_coordinates",
    packages=["xmap_coordinates"],
    package_dir={"xmap_coordinates": "xmap_coordinates"},
    include_package_data=True,
    install_requires=(
        "numpy",
        "xarray",
        "scipy",
    ),
    keywords="xrpattern",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
