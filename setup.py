from setuptools import find_namespace_packages, setup
# works with PEP420 compliant implicit namespace packages

setup(
        name='DeepCV_Lab',
        version=2.0,
        description='Object Detection',
        author='Paul Mc Grath',
        author_email='pmcgrath249@gmail.com',
        url='https://github.com/pmcgrath249/DeepCV_Lab',
        find_namespace_packages(exclude=['data', 'utils.waymo-od*'])  
)