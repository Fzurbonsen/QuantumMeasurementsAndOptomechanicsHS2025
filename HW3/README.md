# Homework 3

## Deps
To run the python script you will need to have the matplotlib, numpy, and pandas libraries installed.
If you don't want to pollute your system simply run it via venv.

## Files
In the `/img` directory you will find the plots which are the solutions to the exercises labeld by their exercise.
To recreate them you have to download the dataset provided with the homework and but the `ix_csv_50x_.csv` file in the `./data/` directory.

## Comments
- In the `main` function you can choose which exercises to run. As later exercises depend on previous exercises you have to run all exercises in the correct order for the first time. This will create three additional files in the `./data/` dircetory. These files are `mean_psd.csv`, `predictive_filter.csv`, and `predictive_filter_mean_psd.csv`. Once these files are created you can run each exercise indiviually by commenting the respective function call in the `main` function.

- To reduce runtime you can omit the loading of certain files at the top of the python script by commenting the relevant lines. Be careful to comment the correct lines depending on which exercise you run. At the beginning of the exercise functions it is stated which files, or rather their respective `DataFrames`, need to be populated for the function to run.
