# Failure Prediction Model 
by Eyal Oren and Mor Hananovitz

## Background / Intro

## Project's goal
To develop a program able to predict when a given physical part (i.e. has its own serial number) will fail quality control tests. The program accomplishes that by generating a forcast of various properties into the future, where each property is allowed to experience changes within given limits. Failure is declared at the moment one of the properties exceeds one of its limits.

## Exploratory Data Analysis
	### Data Structure
	### Let's see it

## Program's Workflow
## Model
LSTM based, multi-step prediction, Currently fixed lengths for input array and for output array.

## Dependencies
	* numpy
	* pandas
	* matplotlib
	* scikit-learn
	* keras 2.x.x
	* tensorflow 1.x.x

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
  - Output: Behavioural prediction in the form of a graph to demonstrate next steps of the part


## How Does the Data Looks Like?

Our data is composed out of 10 parts (Same Make and Model - different Serial Numbers) that ran through the machine in a long period of time (i.e CycleCount).
Each part has 6 features (i.e measured parameters) that define the part's performance in the machine.
Those part (Serial Numbers 1-10) were uses as the base line, training set, to our model's behaviour.

Additional 10 parts were tested (Serial Numbers 11-20) that had low CycleCount, shorter time series, that will supply the testing data to our model. 

## How To Run The Model?
All you have to do is to load attached CSV file and the model will do the rest!
