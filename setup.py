from setuptools import setup, find_packages

def read_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="RL2",
    version="0.1.0",
    packages=find_packages(),
    install_requires=read_requirements('requirements.txt'),
    python_requires='>=3.10',
    author="Chenmien Tan, Simon Yu, Lanbo Lin, Ze Zhang, Yuanwu Xu, Chenhao Jiang, Tianyuan Yang, Sicong Xie, Guannan Zhang",
    author_email="chenmientan@outlook.com",
    description="RL2: Ray Less Reinforcement Learning",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ChenmienTan/RL2"
)