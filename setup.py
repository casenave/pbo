from setuptools import find_namespace_packages, setup

setup(
    name='pbo',
    version='0.0.1',
    packages=find_namespace_packages(include=['pbo', 'pbo.src', 'envs']),
    package_data={'envs': ['*.json']},
    python_requires='>=3.11,<3.14',
    install_requires=[
        'tensorflow==2.20.0',
        'tensorflow-probability==0.25.0',
        'tf-keras==2.20.1',
        'numpy>=1.26,<2.3',
        'matplotlib>=3.9,<4',
        'scipy>=1.14,<2',
    ],
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    entry_points = {
        'console_scripts': ['pbo=pbo.src.main:main'],
    }
)
