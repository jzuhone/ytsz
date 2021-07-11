#!/usr/bin/env python
from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy as np
import os
from pathlib import Path
from subprocess import check_output


def get_szpack_path():
    szpack_path = os.environ.get("SZPACK")
    if szpack_path is None:
        # Guess it's in the HOME dir?
        szpack_path = Path(os.environ["HOME"], "SZpack.v1.1.1")
        if not szpack_path.exists():
            raise IOError("Cannot find your ")
    else:
        szpack_path = Path(szpack_path)
    return szpack_path


szpack_path = get_szpack_path()


gsl_lib_path = Path(check_output(["gsl-config", "--prefix"]).decode().strip(),
                    "lib")

cython_extensions = [
    Extension("ytsz.cszpack",
              ["ytsz/cszpack.pyx"],
              language="c++",
              include_dirs=[".", np.get_include(), szpack_path, szpack_path / "include"],
              library_dirs=[str(szpack_path), str(gsl_lib_path)],
              libraries=["SZpack","gsl"]),
]


setup(name='ytsz',
      packages=find_packages(),
      version="1.0.0",
      description='Python package for simulating SZ observations using yt and SZpack',
      author='John ZuHone',
      author_email='jzuhone@gmail.com',
      url='http://github.com/jzuhone/ytsz',
      setup_requires=["numpy", "cython>=0.24"],
      install_requires=["numpy", "astropy>=4.0", "yt>=4.0.0", "tqdm"],
      include_package_data=True,
      ext_modules=cython_extensions,
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Visualization',
      ],
      )
