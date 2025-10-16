from setuptools import setup, find_packages
setup(
    author='Jolan Lavoisier',
    author_email='jolan.lavoisier@iap.fr',
    description='Classifier of radio traces from cosmic rays and neutrinos using machine learning',
    url='https://github.com/LavoisierJ/CR_radio_classifier#',
    install_requires=[
        'numpy',
        'matplotlib',
        'tensorflow',
        'h5py'
    ],
)