import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='lgndbdt',
    version='0.0.1',
    author='Henry Nachman',
    author_email='henachman@unc.edu',
    description='Python package for extracting PSDs and training a BDT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/henry-e-n/lgndbdt',
    license='Apache-2.0',
    packages=['ML_utils', 'raw_to_bdt', 'extraction_utils'],
    install_requires=['h5py>=3.2.0',
                        'iminuit',
                        'matplotlib',
                        'numpy>=1.21',
                        'pandas',
                        'parse',
                        'pint',
                        'pyfftw',
                        'scipy',
                        'tqdm',
                        'lightgbm']
)
