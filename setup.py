import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='lgndbdt',
    version='0.0.1',
    author='Henry Nachman',
    author_email='henachman@unc.edu',
    description='Python package for extracting PSDs and training a BDT',
    license='Apache-2.0',
    packages=['lgndbdt', 'ML_utils'],
    install_requires=['numpy'],
)
