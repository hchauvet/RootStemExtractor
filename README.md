# RootStemExtractor

Plant motion tracker in python. Extract skeleton of simple rod-shaped object like plants to follow their motions.

This an early public release of the code, a documentation is in preparation.

![Screenshot](https://github.com/hchauvet/RootStemExtractor/raw/master/img/Screenshot1.png "screenshot")

# ChangeLog
* Version 24092018:
  Remove test dectection (Bug with Thread and matplotlib!)
  Correct bug when only one image is loaded, now the processing could be launched.
  
* Version 18052018: 
  Change multiprocessing process (Windows user can now use multiprocessing).
  Change the value of exploration diameter from 0.9 to 1.4 (line 728 of MethodOlivier in new_libgravimacro.py) 
  Remove bug with None values 

# Install

For mac and windows user, the simplest way to install it is to download the Anaconda distribution of python (use the branch 2.x of python)

https://www.continuum.io/downloads

## Python libraries required

Their are all part of the Anaconda python distribution

* pylab (scipy/numpy)
* pandas
* matplotlib
* scikit-image 

* opencv (optional, but increase processing speed)

# Run

## Windows

double click on **RootStemExtractor.bat** file

## On Linux or Mac

open a terminal, go to the RootStemExtractor directory and run `python2 ./RootStemExtractor.py`

### On Mac 

You could try to run the **Mac_RootStemExtractor.app**, it should work...
