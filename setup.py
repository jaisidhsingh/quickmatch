from setuptools import setup, find_packages


setup(
	name="quickmatch",
	version="0.1.0",
	packages=find_packages(),
	install_requires=[
		"numpy==1.26.4",
		"torch==2.2.2",
		"easydict==1.13",
		"facenet-pytorch==2.6.0",
		"tqdm==4.66.4",
		"onedrivedownloader==1.1.3"
	],
	entry_points={
		"console_scripts": [
			"quickmatch = quickmatch.__main__",
		]
	},
)
