from setuptools import setup, find_packages

setup(
    name             = 'mkt',
    version          = '0.1',
    description      = 'Reinforcement Learning from Human (Preference) Feedback for Marketing Phrase Generation',
    author           = 'Dong-Hyun Lee',
    author_email     = 'dhlee347@gmail.com',
    url              = 'https://github.com/frl-lgai/mkt',
    download_url     = 'https://github.com/frl-lgai/mkt/archive/0.1.tar.gz',
    install_requires = [ 
        'torch>=1.8',
        'numpy>=1.20',
        'icecream>=2.1',
        'transformers==4.13.0',
        'datasets>=2.6',
        'omegaconf>=2',
        'wandb>=0.13',
        'typer>=0.6',
    ],
    packages         = find_packages(exclude = ['docs', 'tests*']),
    keywords         = ['deep learning', 'language model', 'gpt', 'reinforcement learning', 'human feedback', 'marketing', 'generation'],
    python_requires  = '>=3.8',
    package_data     =  {},
    zip_safe=False,
)