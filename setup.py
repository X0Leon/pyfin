# -*- coding: utf-8 -*-

import setuptools
import codecs
import os


def local_file(filename):
    return codecs.open(
        os.path.join(os.path.dirname(__file__), filename), 'r', 'utf-8'
    )


setuptools.setup(
    name="pyfin",
    version='1.0.0a',
    author='X0Leon',
    author_email='pku09zl@gmail.com',
    description='Financial toolkit for Quant of China market',
    keywords='python finance quant functions',
    url='https://github.com/X0Leon/pyfin',
    license='MIT',
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'tabulate',
        'matplotlib',
        'scikit-learn',
        'tushare'
    ],
    packages=['pyfin'],
    long_description=local_file('README.md').read(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Software Development :: Libraries',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python',
        # 'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
    ]
)
