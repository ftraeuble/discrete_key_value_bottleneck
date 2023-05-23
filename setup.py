from setuptools import setup

setup(
    name='key_value_bottleneck',
    version='1.0.0',
    description='Python package to provide torch module for the Key Value Bottleneck',
    author='Frederik Träuble, Nasim Rahaman',
    author_email='frederik.traeuble@tuebingen.mpg.de',
    license='MIT License',
    packages=['key_value_bottleneck'],
    install_requires=['torch',
                      'torchvision',
                      'einops',
                      'numpy',
                      'tqdm',
                      'clip @ git+https://github.com/openai/CLIP.git#egg=clip',
                      ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.9.12',
    ],
)
