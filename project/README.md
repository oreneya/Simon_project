# Failure Prediction Model 
by Eyal Oren and Mor Hananovitz

## Project's goal
To develop a program able to predict when a given physical part (i.e. has its own serial number) will fail quality control tests. The program accomplishes that by generating a forecast of various features into the future, where each feature is allowed to experience changes within given limits. Failure is declared at the moment one of the features exceeds one of its spec limits.

## ------------

## Introduction 
In this work we will demonstrate failure prediction model based on previously collected data.
As technology evolves, processes improve, but not everything is changing. We shall take information collected on parts used since 2013 and implement on the same parts used today.

<img src=part.JPG align="center" width=700>
Fig. 1: Part in Question

  - Inputs: Time series (i.e CycleCounts) of the average measured value

<img src=trend_example.png align=“center” width=700>
Fig. 2: Trend Example
  - Features: Parameters collected based on part's manufacture specs
  - Output: Behavioral prediction in the form of a graph to demonstrate next steps of the part
  - Validation: Using a part with long cycle count for validation

<img src=output.png align=“center” width=700>
Fig. 3: Output Example (including the validation series)

## How Does the Data Looks Like?

Our data is composed out of 10 parts (Same Make and Model - different Serial Numbers) that ran through the machine in a long period of time (i.e CycleCount).
Each part has 6 features (i.e measured parameters) that define the part's performance in the machine.
Those part (Serial Numbers 1-10) were uses as the base line, training set, to our model's behavior.

Additional 10 parts were tested (Serial Numbers 11-20) that had low CycleCount, shorter time series that will supply the testing data to our model. 

## Model
LSTM based, multi-step prediction, currently fixed lengths for input array and for output array.

## Dependencies
	* numpy
	* pandas
	* matplotlib
	* scikit-learn
	* keras 2.x.x
	* tensorflow 1.x.x

## Program's Workflow

In current set up the program trains on one feature at a time - the feature in this case is "CDI - Upper Lip Radial Thickness" - found in _main.py_ under the training data (row 47). 
To move between features this must be done manually (full list of features by name is at the end of the README file).

In order to run the model as is - clone the Git repository and run _main.py_ file.

Another possibility is to view all parts used for training (Serial numbers) behavior by feature at the same time - identical to figure 1. 
To do so, also under _main.py_ in the _visualize training data_ section - rows 16, 17, 20, 21, 22 must be unhidden. 

## Features (i.e measured parameters)
Must be copied as is!

	CDI - Flatness
	CDI - Diameter (Inner)- Post Strip
	CDI - Upper Lip Radial Thickness
	CDI - Avg Wall Cell Thickness
	CDI - Miscellaneous- Distance
	Surface Roughness- Post Blast
	Inner Honey Comb Ra (L1 - L6)
	UV Inspection 1=Pass, 0=Fail
	Particle Values ( <1.0 p/cm2 at 0.3um)