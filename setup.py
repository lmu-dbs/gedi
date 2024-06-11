from setuptools import setup, find_packages
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

version_string = os.environ.get("VERSION_PLACEHOLDER", "1.0.0")
print(version_string)
version = version_string

setup(
        name = 'gedi',
        version = str(version),
        description = 'Generating Event Data with Intentional Features for Benchmarking Process Mining',
        author = 'Andrea Maldonado',
        author_email = 'andreamalher.works@gmail.com',
        license = 'MIT',
        url='https://github.com/lmu-dbs/gedi.git',
        long_description=long_description,
        long_description_content_type="text/markdown",
        install_requires=[
            'ConfigSpace==0.7.1',
            'feeed==1.2.0',
            'imblearn==0.0',
            'Levenshtein==0.23.0',
            'matplotlib==3.8.4',
            'numpy==1.26.4',
            'pandas==2.2.2',
            'pm4py==2.7.2',
            'scikit-learn==1.2.2',
            'scipy==1.13.0',
            'seaborn==0.13.2',
            'smac==2.0.2',
            'tqdm==4.65.0',
            ],
        packages = ['gedi'],
        classifiers=[
            'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
            'Intended Audience :: Science/Research',      # Define that your audience are developers
            'Topic :: Software Development',
            'License :: OSI Approved :: MIT License',   # Again, pick a license
            'Programming Language :: Python :: 3.9',
    ],
)