from setuptools import find_packages, setup

setup(name="tensor_rnn",
    version="0.1",
    description="a library for Tensor-based Decomposition for RNN weight parameters",
    author="Andros Tjandra",
    author_email='andros.tjandra@gmail.com',
    platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
    license="MIT",
    url="",
    python_requires='>=3',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'torch', 'scikit-learn', 'requests']);
