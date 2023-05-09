from setuptools import setup, find_packages

setup(
    name='mpl_markers',
    description='interactive marker support for matplotlib',
    author='rlyon14',
    version='0.1.1',
    packages=['mpl_markers',],
    install_requires=(
		'matplotlib>=3.1.3',
        'numpy',
    ),
)
