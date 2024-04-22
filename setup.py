import setuptools
import pathlib 

setuptools.setup(
     name="issim",
     version="0.1.0",
     author="Sean Lewis",
     author_email="sean.lewis@yale.edu",
     description="A Python package for simulating ISM in galaxies",
     long_description= pathlib.Path("README.md").read_text(),
     long_description_content_type = "text/markdown",
     packages=["issim", "issim/galgen"],
     python_requires='>=3',
     url = "https://github.com/dannytalk/ISsiM",
     license = "MIT",
     install_requires=["numpy", "scipy", "matplotlib","astropy", "IPython", "tqdm", "ffmpeg"],
     classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
