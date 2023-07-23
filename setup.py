from setuptools import find_packages, setup
from typing import List

HYPER_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if HYPER_E_DOT in requirements:
            requirements.remove(HYPER_E_DOT)
        return requirements


setup(
    name="Diamond Price Prediction",
    version="0.0.1",
    author="Utpal Paul",
    author_email="utpalpaul108@gmail.com",
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()

)
