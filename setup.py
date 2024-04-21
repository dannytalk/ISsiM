import setuptools

setuptools.setup(
     name="issim",
     version="0.1",
     author="Sean Lewis",
     author_email="sean.lewis@yale.edu",
     description="A Python package for simulating ISM in galaxies",
     packages=["issim", "issim/galgen"],
     python_requires='>=3',
     install_requires=["numpy", "scipy", "matplotlib","astropy", "IPython", "tqdm", "ffmpeg"]
)
