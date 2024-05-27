# python setup.py bdist_wheel

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
        language='c++'
    ),
]

setup(
    name='example_package_liujch1998',
    version='0.0.1',
    author='Jiacheng (Gary) Liu',
    author_email='liujc@cs.washington.edu',
    description='A Python package',
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    install_requires=[
        'pybind11>=2.5.0',
    ],
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
    python_requires='>=3.11',
)
