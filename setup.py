from setuptools import setup, find_packages

setup(
    name="corticod",
    version="0.1.0",
    description="A multimedia codec library to encode and decode data with patterns.",
    author="Ahmet Emin Ünal",
    author_email="aeunal@hotmail.com",
    url="https://github.com/itu-lab/CorticalCodingCodec", 
    packages=find_packages(),
    package_dir= {
        'corticod': 'corticod',
        'corticod.algorithm': 'corticod/algorithm'
    },
    install_requires=[
        "numpy>=2.0.2",
        "pandas>=2.2.3",
        "scikit-learn>=1.6.0",
        "scikit-image>=0.24.0",
        "soundfile>=0.12.1",
        "tqdm>=4.67.1",
        "path",
        "torch",
        "torchvision",
        "torchaudio",
        "torchmetrics>=1.6.1"
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
)
