from setuptools import Extension, setup
import numpy

setup(
	name="fmask",
	ext_modules=[Extension("fmask", ["fmask.c"], include_dirs=[numpy.get_include()])],
)