# python setup.py bdist_wheel
# twine upload --repository testpypi dist/*

# cibuildwheel --output-dir wheelhouse
# twine upload --repository testpypi wheelhouse/*

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the `get_include()`
    method can be invoked. """

    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        'infini_gram.cpp_engine',
        ['infini_gram/cpp_engine.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
        ],
        language='c++',
        extra_compile_args=['-std=c++17'],
    ),
]

setup(
    name='infini_gram',
    version='2.0.0',
    author='Jiacheng (Gary) Liu',
    author_email='liujc@cs.washington.edu',
    description='A Python package for infini-gram',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    # install_requires=[
    #     'pybind11>=2.5.0',
    # ],
    packages=setuptools.find_packages(),
    # package_data={
    #     'infini_gram': ['engine.py', '*.so'],
    # },
    # classifiers=[
    #     'Programming Language :: Python :: 3',
    #     'Programming Language :: Python :: 3.7',
    #     'Programming Language :: Python :: 3.8',
    #     'Programming Language :: Python :: 3.9',
    #     'Programming Language :: Python :: 3.10',
    #     'Programming Language :: C++',
    # ],
    # license_files=['LICENSE'],
    # include_package_data=True,
    license='UW Academic Software License',
    python_requires='>=3.8',
)
