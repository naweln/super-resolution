
from setuptools import setup, find_packages

packages = find_packages(exclude=('super_resolution.tests*', 'super_resolution.*.tests*'))

setup(
	name="super_resolution",
	version="9999",
	description = "Data processing tools for optoacoustic super resolution",
	author = "Nawel Naas and Berkan Lafci",
	author_email = "naasn@student.ethz.ch and lafciberkan@gmail.com",
	keywords = ["optoacoustic", "photoacoustic", "image reconstruction", "data analysis"],
	classifiers = [],
	install_requires = [],
	provides = ["super_resolution"],
	packages = packages,
	include_package_data=True,
	extras_require = {},
	entry_points = {},
	)
