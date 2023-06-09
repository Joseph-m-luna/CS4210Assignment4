#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #4
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import csv

dbTraining = []
dbTest = []
X_training = []
y_training = []
classVotes = [] #this array will be used to count the votes of each classifier

total = 0
correct = 0

#reading the training data from a csv file and populate dbTraining
with open('optdigits.tra', 'r') as f:
   read = csv.reader(f, delimiter=',')
   for row in read:
      intRow = []
      for item in row:
         intRow.append(int(item))
      dbTraining.append(intRow)

#reading the test data from a csv file and populate dbTest
with open('optdigits.tes', 'r') as f:
   read = csv.reader(f, delimiter=',')
   for row in read:
      intRow = []
      for item in row:
         intRow.append(int(item))
      dbTest.append(intRow)

#inititalizing the class votes for each test sample. Example: classVotes.append([0,0,0,0,0,0,0,0,0,0])
nSamples = len(dbTest)
for n in range(0, nSamples):
   classVotes.append([0,0,0,0,0,0,0,0,0,0])

print("Started my base and ensemble classifier ...")

for k in range(20): #we will create 20 bootstrap samples here (k = 20). One classifier will be created for each bootstrap sample

  bootstrapSample = resample(dbTraining, n_samples=len(dbTraining), replace=True)

  #populate the values of X_training and y_training by using the bootstrapSample
  X_training = []
  y_training = []
  for sample in bootstrapSample:
     X_training.append(sample[0:64])
     y_training.append(sample[-1])
     

  #fitting the decision tree to the data
  clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=None) #we will use a single decision tree without pruning it
  clf = clf.fit(X_training, y_training)
  
  total = 0
  correct = 0
  for i, testSample in enumerate(dbTest):

      #make the classifier prediction for each test sample and update the corresponding index value in classVotes. For instance,
      # if your first base classifier predicted 2 for the first test sample, then classVotes[0,0,0,0,0,0,0,0,0,0] will change to classVotes[0,0,1,0,0,0,0,0,0,0].
      # Later, if your second base classifier predicted 3 for the first test sample, then classVotes[0,0,1,0,0,0,0,0,0,0] will change to classVotes[0,0,1,1,0,0,0,0,0,0]
      # Later, if your third base classifier predicted 3 for the first test sample, then classVotes[0,0,1,1,0,0,0,0,0,0] will change to classVotes[0,0,1,2,0,0,0,0,0,0]
      # this array will consolidate the votes of all classifier for all test samples
      prediction = clf.predict([testSample[0:64]])
      classVotes[i][prediction[0]] += 1

      
      if k == 0: #for only the first base classifier, compare the prediction with the true label of the test sample here to start calculating its accuracy
         total += 1
         if prediction[0] == testSample[-1]:
            correct += 1

  if k == 0: #for only the first base classifier, print its accuracy here
     accuracy = correct/total
     print("Finished my base classifier (fast but relatively low accuracy) ...")
     print("My base classifier accuracy: " + str(accuracy))
     print("")

  #now, compare the final ensemble prediction (majority vote in classVotes) for each test sample with the ground truth label to calculate the accuracy of the ensemble classifier (all base classifiers together)
result = []

for x in classVotes:
   maxInd = 0
   maxVal = 0
   for i, y in enumerate(x):
      if y > maxVal:
         maxVal = y
         maxInd = i
   result.append(maxInd)

correct = 0
total = 0
for i, n in enumerate(result):
   total += 1
   if n == dbTest[i][-1]:
      correct += 1

accuracy = correct/total
     

#printing the ensemble accuracy here
print("Finished my ensemble classifier (slow but higher accuracy) ...")
print("My ensemble accuracy: " + str(accuracy))
print("")

print("Started Random Forest algorithm ...")

#Create a Random Forest Classifier
clf=RandomForestClassifier(n_estimators=20) #this is the number of decision trees that will be generated by Random Forest. The sample of the ensemble method used before

#Fit Random Forest to the training data
clf.fit(X_training,y_training)

#make the Random Forest prediction for each test sample. Example: class_predicted_rf = clf.predict([[3, 1, 2, 1, ...]]
total = 0
correct = 0
for i in dbTest:
   total += 1
   prediction = clf.predict([i[0:64]])

   #compare the Random Forest prediction for each test sample with the ground truth label to calculate its accuracy
   if prediction == i[-1]:
      correct += 1

#printing Random Forest accuracy here
print("Random Forest accuracy: " + str(accuracy))

print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")
