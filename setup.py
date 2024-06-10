from setuptools import setup, find_packages


setup(
	name="quickmatch",
	version="0.1.0",
	packages=find_packages(),
	entry_points={
		"console_scripts": [
			"quickmatch = quickmatch.__main__",
		]
	},
)
