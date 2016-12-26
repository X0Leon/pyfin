# -*- coding: utf-8 -*-

import setuptools


setuptools.setup(
    name="pyfin",
    version='1.0.0a1',
    author='Leon Zhang',
    author_email='pku09zl@gmail.com',
    description='Financial toolkit for Quant',
    keywords='python finance quant toolkit',
    url='https://github.com/X0Leon/pyfin',
    license='MIT',
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'tabulate',
        'matplotlib',
        'scikit-learn',
        'requests'
    ],
    packages=['pyfin'],
    long_description='Financial toolkit for Quantitative investment of Chinese market built on pandas.',
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
