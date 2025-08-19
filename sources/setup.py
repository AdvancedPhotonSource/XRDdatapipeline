from distutils.core import setup, Extension

setup(name="fmask",ext_modules=[Extension('fmask',['fmask.c'])])