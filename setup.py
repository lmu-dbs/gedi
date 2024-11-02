from setuptools import setup, find_packages
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

version_string = os.environ.get("VERSION_PLACEHOLDER", "0.0.10")
print(version_string)
version = version_string

setup(
        name = 'gedi',
        version = str(version),
        packages=find_packages(),
        description = 'Generating Event Data with Intentional Features for Benchmarking Process Mining',
        author = 'Andrea Maldonado',
        author_email = 'andreamalher.works@gmail.com',
        license = 'MIT',
        url='https://github.com/lmu-dbs/gedi.git',
        long_description=long_description,
        long_description_content_type="text/markdown",
        include_package_data=True,
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
            'streamlit-toggle-switch>=1.0.2',
            'click==8.1.7',
            'cloudpickle==3.0.0',
            'configspace==0.7.1',
            'cvxopt==1.3.2',
            'dask==2024.2.1',
            'dask-jobqueue==0.8.5',
            'deprecation==2.1.0',
            'distributed==2024.2.1',
            'emcee==3.1.4',
            'feeed == 1.2.0',
            'fsspec==2024.2.0',
            'imbalanced-learn==0.12.0',
            'imblearn==0.0',
            'importlib-metadata==7.0.1',
            'intervaltree==3.1.0',
            'jinja2==3.1.3',
            'levenshtein==0.23.0',
            'locket==1.0.0',
            'lxml==5.1.0',
            'markupsafe==2.1.5',
            'more-itertools==10.2.0',
            'msgpack==1.0.8',
            'networkx==3.2.1',
            'numpy==1.26.4',
            'pandas>=2.0.0',
            'partd==1.4.1',
            'pm4py==2.7.2',
            'psutil==5.9.8',
            'pydotplus==2.0.2',
            'pynisher==1.0.10',
            'pyrfr==0.9.0',
            'pyyaml==6.0.1',
            'rapidfuzz==3.6.1',
            'regex==2023.12.25',
            'scikit-learn==1.2.2',
            'seaborn==0.13.2',
            'smac==2.0.2',
            'sortedcontainers==2.4.0',
            'stringdist==1.0.9',
            'tblib==3.0.0',
            'toolz==0.12.1',
            'tqdm==4.65.0',
            'typing-extensions==4.10.0',
            'urllib3==2.2.1',
            'zict==3.0.0'
            ],
        classifiers=[
            'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
            'Intended Audience :: Science/Research',      # Define that your audience are developers
            'Topic :: Software Development',
            'License :: OSI Approved :: MIT License',   # Again, pick a license
            'Programming Language :: Python :: 3.9',
    ],
)
