from setuptools import setup, find_packages

setup(
    name='utils',
    version='0.1.0',
    author='Noel Kronenberg',
    author_email='noel.kronenberg@charite.de',
    description='Data science utilities.',
    url='https://github.com/noelkronenberg/utils',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # data handling
        'pandas',
        'numpy',
        # visualization
        'matplotlib',
        'seaborn',
        'plotly',
        # analysis
        'scikit-learn',
        'statsmodels',
        'stepmix'
    ],
)