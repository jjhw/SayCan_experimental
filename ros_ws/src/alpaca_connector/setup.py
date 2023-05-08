from setuptools import setup

setup(
    name="alpaca_connector",
    version="0.0.1",
    description="Python library to connect to Alpaca stand",
    author="Yaroslav Savelev",
    author_email="yar21sav@gmail.com",
    packages=["alpaca_connector"],
    install_requires=["rospy", "opencv-contrib-python"],
    license="GNU Lesser General Public License v3",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Topic :: System :: Hardware :: Hardware Drivers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ])
