from setuptools import setup, find_packages

setup(
    name="emotion_analysis",
    version="0.1.0",
    description="A system to analyze emotions in video, capturing moments of laughter, neutral, and sadness.",
    author="Juan Manoel",
    author_email="juanengml@gmail.com",
    url="https://github.com/juanengml/Video-Emotion-Analysis-System",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "flask-opencv-streamer",
        "ultralytics",
        "deepface",
        "bbox-visualizer",
        "prettytable",
        "flask"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
