from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="cmm",
    version="0.1",
    description="Coherence Mixture Model",
    url="https://github.com/vrutten/cmm",
    author="Virginia MS Rutten",
    author_email="ruttenv@janelia.hhmi.org",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=["src"],
    install_requires=["numpy", "jax", "tqdm"],
)
