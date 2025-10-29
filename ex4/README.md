# Exercise 4: "Power spectral density of random telegraph noise"
This directory hold my solutions to exercise 4 parts c) - e).

## How to run the code:

The code is written in python and uses specific libraries. To run the code use venv.

### How to load requirments
To use the requirements execute the following commands:
````
python -m venv .venv
source .venv/bin/activate       # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```` 

### How to update requirements
To update the requirements execute the following commands:
````
pyhton -m venv .venv
source .venv/bin/activate       # or .venv\Scripts\activate on Windows
# change requirements:
pip install pandas numpy etc.

# update requirements.txt
pip freeze > requirements.txt
````

### How to exit venv
To exit venv you can type `deactivate`