import os
import sys
from setuptools import setup, Extension

with open(os.path.join(os.path.dirname(__file__), 'README.rst')) as readme:
	README = readme.read()

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

tree = Extension('zifor.tree',
	sources = ['src/tree.cpp'],
	language = 'c++',
	extra_compile_args = ['-std=c++17'],
	undef_macros = [ "NDEBUG" ],
	libraries = ['boost_python-py3', 'boost_numpy-py3'])

setup(
	name='zifor',
	version='0.1.0',
	packages=['zifor'],
	ext_modules=[tree],
	include_package_data=True,
	license='GPL-3.0-or-later',
	description='A description',
	long_description=README,
	url='https://github.com/matwey/zifor',
	author='Matwey V. Kornilov',
	author_email='matwey.kornilov@gmail.com',
	classifiers=[
		'Programming Language :: Python',
		'Programming Language :: Python :: 3',
	]
)
