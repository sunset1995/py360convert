from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='py360convert',
    packages=['py360convert'],
    version='0.1.0',
    license='MIT',
    description='Convertion between cubemap and equirectangular and also to perspective planar.',
    long_description=long_description,
    author='Cheng Sun',
    author_email='chengsun@gapp.nthu.edu.tw',
    url='https://github.com/sunset1995',
    download_url='https://github.com/sunset1995/py360convert/archive/v_0.1.0.tar.gz',
    install_requires=[
        'numpy',
        'scipy',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Artistic Software',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    scripts=['convert360'],
)
