"""
SEVA Agent - 自驱永续进化预测修正Agent系统
打包配置文件
"""
from setuptools import setup, find_packages
import os

# 读取README
here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, "README.md")
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

# 读取依赖
requirements_path = os.path.join(here, "requirements.txt")
install_requires = []
if os.path.exists(requirements_path):
    with open(requirements_path, "r", encoding="utf-8") as f:
        install_requires = [
            line.strip() for line in f
            if line.strip() and not line.startswith("#")
        ]

# 读取版本号
version = "0.40.0"
init_path = os.path.join(here, "seva", "__init__.py")
if os.path.exists(init_path):
    with open(init_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().strip('"').strip("'")
                break

setup(
    name="seva-agent",
    version=version,
    description="自驱永续进化预测修正Agent系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SEVA Team",
    author_email="2358384171@qq.com",
    url="https://github.com/17683995446/seva-agent",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*", "examples",
                                    "examples.*"]),
    python_requires=">=3.8",
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "seva=seva.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="agent, ai, self-evolving, prediction, reinforcement-learning",
    include_package_data=True,
    zip_safe=False,
)
