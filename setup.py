from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="veronica",
    version="2.5.0",
    author="Barzan DIG",
    author_email="basel.anay@barzn.com",
    description="A secure face authentication system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Basel-anaya/veronica",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Security :: Authentication",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "veronica": [
            "web/static/css/*.css",
            "web/static/js/*.js",
            "web/templates/*.html",
        ],
    },
    entry_points={
        "console_scripts": [
            "veronica=veronica.api.routes:main",
        ],
    },
) 