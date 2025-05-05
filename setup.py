import os, re
from setuptools import setup, find_packages

version_file=os.path.join('jaxpt','info.py')
verstrline = open(version_file, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    jaxpt_version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (version_file,))
print('FASTPT version is %s'%(jaxpt_version))

setup(version=jaxpt_version, packages=find_packages())