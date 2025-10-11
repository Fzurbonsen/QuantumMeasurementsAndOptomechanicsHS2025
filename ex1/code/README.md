## USE VENV TO RUN THE SCRIPT

### How to load requirements
To use the requirements execute the following commands:
````
python -m venv .venv
source .venv/bin/activate # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```` 

### How to update requirements
To update the requirements execute the following commands:
````
pyhtoin -m venv .venv
source .venv/bin/activate # or .venv\Scripts\activate on Windows
# change requirements:
pip install pandas numpy etc.

# update requirements.txt
pip freeze > requirements.txt
````

### How to exit venv
To exit venv you can type `deactivate`