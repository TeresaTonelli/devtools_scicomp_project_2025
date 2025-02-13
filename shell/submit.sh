
#!/bin/bash

conda create --name devtools_scicomp python=3.9
conda activate devtools_scicomp

python -m pip install pytest

REPO="devtools_scicomp_project_2025"

cd devtools_scicomp_teresa
git clone git@github.com:TeresaTonelli/${REPO}.git 
cd devtools_scicomp_project_2025/

DESC="Project for the creation of a github repository"
LIC="License"
NAME="Teresa"
EMAIL="teresatonelli99"

#README creation
nano README.md
git add .
git commit -m "first commit"
git push origin HEAD:main

#Structuring the package
mkdir src/pyclassify/
mkdir scripts
mkdir test
mkdir shell
mkdir experiments

touch src/pyclassify/__init__.py
touch src/pyclassify/utils.py
touch scripts/run.py
touch shell/submit.sbatch 
touch  shell/submit.sh
touch experiments/config.yaml
touch test/test_.py

#generate requirements.txt
python -m pip freeze > requirements.txt

#create the pyproject.toml 
touch pyproject.toml
wget 'https://github.com/dario-coscia/devtools_scicomp_project_2025/blob/main/pyproject.toml' -O pyproject.toml
sed -i 's/description = "INSERT"/description = "${DESC}"/' pyproject.toml
sed -i 's/file = "LICENSE"/file = ${LIC}/' pyproject.toml
sed -i 's/name = "INSERT"/name = "${NAME}"/' pyproject.toml
sed -i 's/email = "INSERT@gmail.com"/email = "{$EMAIL}@gmail.com/' pyproject.toml

#modify the .gitignore file
echo "'#Ignore .dat and .data files'" >> .gitignore
echo ".dat" >> .gitignore
echo ".data" >> .gitignore

#add, commit and push
git add .
git commit -m "structuring the package"
git push origin HEAD:main
