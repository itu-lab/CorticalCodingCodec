from setuptools import setup, find_packages

setup(
    name="corticod",
    version="0.1.0",
    description="A biomimetic cortical coding library for audio compression and signal processing.",
    author="Ahmet Emin Ünal",
    author_email="aeunal@hotmail.com",
    url="https://github.com/itu-lab/CorticalCodingCodec", 
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",  
    packages=find_packages(),
    package_dir= {
        'corticod': 'corticod',
        'corticod.algorithm': 'corticod/algorithm',
        'corticod.utils': 'corticod/utils'
    },
    install_requires=[
        "numpy>=2.0.2",
        "pandas>=2.2.3",
        "scikit-learn>=1.6.0",
        "scikit-image>=0.24.0",
        "soundfile>=0.12.1",
        "path",
        "torch",
        "torchvision",
        "torchaudio",
        "torchmetrics>=1.6.1",
        "scikit-image",
        "constriction"
    ],
    dependency_links=[
        'https://download.pytorch.org/whl/cu118/'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    include_package_data=True  # Includes non-code files like README
)
