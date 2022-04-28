import setuptools

# Fetch ReadMe
with open("README.md", "r") as fh:
    long_description = fh.read()

# Use requirements.txt to set the install_requires
with open('requirements.txt') as f:
    install_requires = [line.strip() for line in f]

setuptools.setup(
    name="visualize-events",
    version="v0.0",
    author="Sean McCarren",
    author_email="mccarrensean at gmail dot com",
    description="Visualizing adverse events of drug treatments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SeanMcCarren/visualizing-side-effects",
    packages=setuptools.find_packages(),
    python_requires='>=3.8',
    install_requires=install_requires
)