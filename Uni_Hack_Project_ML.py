# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:02:55 2024

@author: Zenan Chen
"""

# Note: we are more competent in using pylab than numpy,
# so we decided to opt for pylab in this project.
import pylab

# Euclidean metric used for k-nearest neighbours.
def euclideanDist(list1, list2):
    '''
    Assumes list1 and list2 are lists of numbers.
    
    Returns the Euclidean distance for the two
    lists, as a pylab array.
    '''
    array1 = pylab.array(list1[:4])
    array2 = pylab.array(list2[:4])
    return((sum(list(map(lambda x: x**2, array1 - array2))))**0.5)


# Scaling to ensure all variables are of equal weight.
# Z-scaling is chosen here, although linear scaling could
# also work.
def ZscaledData(data):
    '''
    Assumes data is a list of lists.
    
    Returns the Z-scaled version of each list,
    as well as the mean and standard deviation (stdv),
    which is useful in scaling the input_data
    later on.
    '''
    # Find mean.
    data_array = []
    for lst in data:
        data_array.append(pylab.array(lst[:4]))
    mean = sum(data_array)/len(data_array)
    
    # Find standard deviation (stdv).
    diff_sqr = pylab.array([0,0,0,0])
    diff = []
    for array in data_array:
        arr = array[:] - mean
        diff.append(arr)
        diff_sqr = diff_sqr + pylab.array(list(map(lambda x: x**2, arr)))
    stdv = (diff_sqr/len(data_array))**0.5
    
    # Standardise.
    standard_data = list(diff/stdv)
    final_output =[]
    for elem in standard_data:
        final_output.append(list(elem))
        
    # Add the label (final element) back in and return.
    for element in range(len(data)):
        final_output[element].append(data[element][-1])
    return (final_output, list(mean), list(stdv))


# This is the data set the machine learning algorithm
# is learned on. We could've stored this in a separate .txt
# file, but we weren't sure if we are allowed to submit multiple
# files in this competition.
TRAINING_DATA = [
[4,7,27,50,'Date Palms'],
[4.25,7.05,27.4,51.5,'Date Palms'],
[4.5,7.1,27.8,53,'Date Palms'],
[4.75,7.15,28.2,54.5,'Date Palms'],
[5,7.2,28.6,56,'Date Palms'],
[5.25,7.25,29,57.5,'Date Palms'],
[5.5,7.3,29.4,59,'Date Palms'],
[5.75,7.35,29.8,60.5,'Date Palms'],
[6,7.4,30.2,62,'Date Palms'],
[6.25,7.45,30.6,63.5,'Date Palms'],
[6.5,7.5,31,65,'Date Palms'],
[6.75,7.55,31.4,66.5,'Date Palms'],
[7,7.6,31.8,68,'Date Palms'],
[7.25,7.65,32.2,69.5,'Date Palms'],
[7.5,7.7,32.6,71,'Date Palms'],
[7.75,7.75,33,72.5,'Date Palms'],
[8,7.8,33.4,74,'Date Palms'],
[8.25,7.85,33.8,75.5,'Date Palms'],
[8.5,7.9,34.2,77,'Date Palms'],
[8.75,7.95,34.6,78.5,'Date Palms'],
[9,8,35,80,'Date Palms'],
[1.5,6.5,10,120,'Watercress'],
[1.555,6.525,10.25,122.5,'Watercress'],
[1.61,6.55,10.5,125,'Watercress'],
[1.665,6.575,10.75,127.5,'Watercress'],
[1.72,6.6,11,130,'Watercress'],
[1.775,6.625,11.25,132.5,'Watercress'],
[1.83, 6.65,11.5,135,'Watercress'],
[1.885, 6.675,11.75,137.5,'Watercress'],
[1.94, 6.7,12, 140,'Watercress'],
[1.995, 6.725,12.25, 142.5,'Watercress'],
[2.05, 6.75,12.5,145,'Watercress'],
[2.105, 6.775,12.75,147.5,'Watercress'],
[2.16, 6.8,13,150,'Watercress'],
[2.215, 6.825,13.25,152.5,'Watercress'],
[2.27, 6.85, 13.5,155,'Watercress'],
[2.325, 6.875, 13.75,157.5,'Watercress'],
[2.38, 6.9, 14,160,'Watercress'],
[2.435, 6.925, 14.25,162.5,'Watercress'],
[2.49, 6.95, 14.5,165,'Watercress'],
[2.545, 6.975,14.75,167.5,'Watercress'],
[2.6, 7,15, 170,'Watercress']
]

# The machine learning algorithm used is k-nearest neighbours.
# In the future, we can consider implementing a different algorithm
# such as logisitc regression, as well as using different values of k.
# In this algorithm, k = 5 often lead to optimal results.
def KNN_Implementation(training_data, input_data):
    '''
    Assumes training_data a list of lists, and input_data
    a list.
    
    Uses k-nearest neighbours to determine whether Watercress
    or Date Palm is more suitable given the water and temperature
    conditions, for k = 5.
    
    Credits to:
    Guttag, J.V. (2017) ‘CHAPTER 24. CLASSIFICATION METHODS’, 
    in Introduction to Computation and Programming Using Python with 
    Application to Understanding Data. 2nd edn. Cambridge, Massachusetts: 
    The MIT Press, pp. 403–411. 
    '''
    # Standardise training and input data.
    standardised_training_data, mean, stdv = ZscaledData(training_data)
    scaled_input = (pylab.array(input_data) - mean)/stdv
    
    # Initialise variables and distances.
    KNN, distances = [scaled_input], [0]
    for _i in range(5):
        KNN.append(standardised_training_data[_i])
        distances.append(euclideanDist(standardised_training_data[_i]
                                       , scaled_input))
    max_dist = max(distances)
    
    # Replace the maximum distance if the distance of one of
    # the other data points in the training data is less
    # than max_dist.
    for condition in standardised_training_data[5:]:
        dist = euclideanDist(condition, scaled_input)
        if dist < max_dist:
            max_index = distances.index(max_dist)
            KNN[max_index] = condition
            distances[max_index] = dist
            max_dist = max(distances) 
    
    # Construct a 'voting system' which picks out the
    # 5 nearest neighbours and records which plant it is.
    frequency_dict = {}
    for plant in KNN[1:]:
        try:
            frequency_dict[plant[-1]] += 1
        except KeyError:
            frequency_dict[plant[-1]] = 1
    max_freq = max(frequency_dict.values())
    for key in frequency_dict:
        if frequency_dict[key] == max_freq:
            # Return the 'majority vote'.
            return(f'The most suitable plant is {key.lower()}.')
        
        
# Line 172 prevents this from being executed if this code was 
# ever imported as a module.
if __name__ == '__main__':
    
    # Intialise input data point.
    salinity = float(input('Enter salinity of water (in dS/m): '))
    pH = float(input('Enter pH of water: '))
    temperature = float(input('Enter temperature of \
environment (in °C): '))
    concentration = float(input('Enter concentration \
of Ca²⁺ in water (in mg/100g water): '))
    inp = [salinity, pH, temperature, concentration]
    
    # Call and print result of KNN.
    print(KNN_Implementation(TRAINING_DATA, inp))