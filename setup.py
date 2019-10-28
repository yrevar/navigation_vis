import setuptools
# versioning style inspired from https://github.com/david-abel/simple_rl/
exec(open('navigation_vis/_version.py').read())

# Ref: https://packaging.python.org/tutorials/packaging-projects/

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="navigation_vis",
    version=__version__,
    author="Yagnesh Revar",
    author_email="mailto.yagnesh+github@gmail.com",
    description="A lightweight library for visualizing navigation grid world",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yrevar/navigation_vis",
    packages=setuptools.find_packages(),
    keywords = ['Markov Decision Process', 'MDP', 'Navigation', 'Data Visualization'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)