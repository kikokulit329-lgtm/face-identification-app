from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="face-identification-app",
    version="1.0.0",
    author="Your Name",
    description="A real-time face identification application using computer vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kikokulit329-lgtm/face-identification-app",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.7",
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "tensorflow>=2.13.0",
        "face-recognition>=1.3.5",
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
    ],
)
