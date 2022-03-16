from setuptools import setup

setup(
    name='abstract_broker',
    version='0.0.1',
    packages=['abstract_broker'],
    url='',
    license='',
    author='bjahnke',
    author_email='bjahnke71@gmail.com',
    description='Interfaces for normalizing broker client/stream APIs',
    install_requires=[
        "numpy",
        "pandas",
    ]
)
