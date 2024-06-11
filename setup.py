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
    author='Jaisidh Singh',
    author_email='jaisidhsingh@gmail.com',
    description='A command-line tool for obtaining face-matcher embeddings across 4 SOTA face-matchers.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/jaisidhsingh/quickmatch',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
