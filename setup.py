# Run this script from the origin folder as:
#   > "python setup.py clean" in order to clean previous builds
#   > "python setup.py test" in order to execute all the unittests
#   > "python setup.py sdist" in order to build the library
#
# The package can then be published with:
#   > twine upload dist/*

from setuptools import find_packages, setup

# set up the library metadata and make the build
with open('README.md', 'r') as readme:
    setup(
        name='covid data analysis',
        version='0.0.1',
        maintainer='Federico Baldo',
        maintainer_email='federico.baldo2@unibo.it',
        author='Federico Baldo',
        long_description_content_type="text/markdown",
        packages=find_packages(),
        python_requires='>=3.9'
    )