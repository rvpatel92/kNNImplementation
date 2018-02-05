import numpy

#calculate if algorithm is correct or wrong
def knn(trainData, testData, kInput):
    predictionCorrect = 0
    predictionWrong = 0
    i = 0
    classification = map(lambda d: knnImplementation(trainData, d, kInput), testData)
    for specificClassification in classification:
        if testData[i][0] == (int)(specificClassification):
            predictionCorrect += 1
        else:
            predictionWrong += 1
        i+=1

    accuracy = algorithmEfficiency(predictionCorrect, predictionWrong)
    return accuracy

def knnImplementation(trainData, d, kInput):
    distanceArray = []
    numberOfOccurances = numpy.zeros(10)

    for tData in trainData:
        distance = 0;
        # Calculate distance between the query-instance and all the training examples
        for i in range(1, len(tData)):
            distance += pow((tData[i] - d[i]), 2)
        distance = numpy.math.sqrt(distance)
        convertToList = [distance, tData[0]]
        distanceArray.append(convertToList)

    # Sort the distance and determine nearest neighbors based on the k-th minimum distance
    sortedDistanceArray = sorted(distanceArray)

    for i in range(0, kInput):
        test = sortedDistanceArray[i][1]
        numberOfOccurances[(int)(test)] += 1
    # return the max number of occurances through 0-9 to determine which class it belongs to
    return numberOfOccurances.argmax()

# Calculate algorithm accuracy
def algorithmEfficiency(predictionCorrect, predictionWrong):
    accuracy = (float)(predictionCorrect) / (float)((predictionCorrect + predictionWrong))
    return accuracy

# Using numpy to read csv files / remove headers / ask for kInput / run with algorithm
trainData = numpy.genfromtxt('MNIST_training.csv', delimiter=',', skip_header=1)
testData = numpy.genfromtxt('MNIST_test.csv', delimiter=',', skip_header=1)
kInput = input('What value would you like to give K: ')
print('Running algorithm...')

accuracy = knn(trainData, testData, kInput)

print('Accuracy of the algorithm is: ')
print'{percent:.2%}'.format(percent=accuracy)