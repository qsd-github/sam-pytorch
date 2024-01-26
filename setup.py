from setuptools import setup

setup(
    name='sam-pytorch',
    version='1.0',
    description='The SAM model using pytorch for inference',
    author='qsd',
    author_email='2472645980@qq.com',
    packages=["sam", "sam.utils", "sam.utils.display", "sam.utils.url"],
    install_requires=[
        "numpy==1.24.4",
        "openai-clip==1.0.1",
        "opencv-python==4.5.5.62",
        "requests==2.28.1",
        "segment-anything==1.0",
        "matplotlib==3.7.0",
        "tqdm==4.64.1",
        "torch==2.0.1+cu117",
        "torchvision==0.15.2+cu117"
    ],
)
