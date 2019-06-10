import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="concrete-autoencoder",
    version="0.0.1",
    author="Muhammed Fatih Balin",
    author_email="m.f.balin@gmail.com",
    description="An implementation of Concrete Autoencoders",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mfbalin/Concrete-Autoencoders",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['keras'],
    python_requires='>=3',
)
