#! /usr/bin/env python
"""Tools for PySpark and Databricks."""

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join("psutils", "_version.py")
with open(ver_file) as f:
    exec(f.read())

DISTNAME = "pandas_extensions"
DESCRIPTION = "Utilities for PySpark and Databricks"
with codecs.open("README.rst", encoding="utf-8-sig") as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = "A. Lieberman"
MAINTAINER_EMAIL = "avi.lieberman@ey.com"
URL = "https://github.com/avilieber/pandas_accessors"
LICENSE = "new BSD"
DOWNLOAD_URL = "https://github.com/avilieber/pandas_accessors"
VERSION = __version__  # noqa: F821
with open("requirements.txt", "r") as f:
    INSTALL_REQUIRES = [r.strip() for r in f.readlines() if not r.startswith("#")]
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.7",
]
EXTRAS_REQUIRE = {
    "tests": ["pytest", "pytest-cov"],
    "docs": ["sphinx", "sphinx-gallery", "sphinx_rtd_theme", "numpydoc", "matplotlib"],
}

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
