from setuptools import find_packages,setup
from typing import List

hypen='-e .'
def get_requires(require:str)->List[str]:
    requirements=[]
    with open(require) as file_obj:
        requirements=file_obj.readlines()
        requirements=[i.replace('/n',' ') for i in requirements]
        if hypen in requirements:
            requirements.remove(hypen)



setup(
    author='manoj',
    name='mlproc',
    version='0.0.1',
    author_email='manojthupakula06080@gmail',
    packages=find_packages(),
    install_requires=get_requires('requirements.txt')
    
)