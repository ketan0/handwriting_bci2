from setuptools import setup, find_packages

setup(
    name='neural_decoder',
    version='0.0.1',
    packages=find_packages(include=['neuralDecoder']),
    install_requires=[
        'tensorflow~=2.9.0',
        'hydra-core==1.1.0',
        'hydra-submitit-launcher==1.1.5',
        'transformers==4.12.5',
        'redis',
        'seaborn',
        'pandas',
        'jupyterlab',
        'ipywidgets',
        'tqdm',
        'wandb',
        'g2p_en==2.0.0'
    ]
)
