Author: Benjamin Lanoue

This project will use logistic regression through gradient descent on a set of data.
This will produce an algorithm to calculate whether a breast tumor is malignant or benign.
It will do this by outputting an array 1D array which represents theta values. 
Multiplying these theta values to the corresponding input values and summing them all will output
a value which will give the most likely scenerio for the input values, either malignent or benign.
The following data comes from here: https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)
The description of data is as follows:

------------------------------------------------------------------------------------------

Data Set Information:

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. A few of the images can be found at [Web Link]

Separating plane described above was obtained using Multisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree Construction Via Linear Programming." 
Proceedings of the 4th Midwest Artificial Intelligence and Cognitive Science Society, pp. 97-101, 1992], 
a classification method which uses linear programming to construct a decision tree. Relevant features were selected using an exhaustive search in the space of 1-4 features and 1-3 separating planes.

The actual linear program used to obtain the separating plane in the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

This database is also available through the UW CS ftp server:
ftp ftp.cs.wisc.edu
cd math-prog/cpo-dataset/machine-learn/WDBC/


Attribute Information:

1) ID number
2) Diagnosis (M = malignant, B = benign)
3-32

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)



------------------------------------------------------------------------------------------
DATA

Column 0 distinguishes a unique row
Column 1 describes M for malignant, and B for benign; This will be the output column (y values)
Columns 2-31 are various numerical data.
There are no missing values in the data set


PREPROCESSING DATA

The first thing needed for this project is to preprocess the data so that the regression can
1. Run properly (make data readable to the machine)
2. Run efficiently (make it so we can minimize T)

First, read in the data as a pandas object, and delete the [0] column.
This column is used as a unique identifier and will not
be needed. This also slightly shrinks the size of the data.
To make this data readable we can change the values of the new Column[0] to 1s and 0s.
If a row is Malignent, it will be 1
If a row is Benign, it will be 0
With this, I will store the result in a new array called 'y', which will stand for the
dependent variable/output array, then drop the row from wdbc, and set the new df as 'x'

Next I will drop the column [0] and store it in a seperate variable.
This new 1D pandas dataframe will be the output array.
The old pandas dataframe will no only consist of independent variables,
which can then be featured scaled.


Next mean normalization will be performed on the independent variable df.
Normalizedx = (x-mean)/[max(x)-min(x)]


HYPOTHESIS

Now that we have a 2D array with only preprocessed 
independent variables (called x_i), we can calculate a hypothesis.
The hypothesis will be calculated with a 1D array theta[], and 1 row of x_i.
theta[] will be the amount of columns/attributes + 1 in x_i (x_i+1)

i= Current row of x
j= Currnet column of x and theta where j is the size of len(theta[])
hypothesis_x = theta[0] + theta[1]*x_i[i][0] + theta[2]*x_i[i][1] + .... theta[j]*x_i[i][j-1]

The value of hypothesis_x (h_x) will be put in to the sigmoid function to more properly
calculate the chance of a tumor being benign or malignent. 

Chance of malignency = 1/(1+e^[-h_x])

where is e = euler's number


COST

The cost function for linear regression wont work for logistic regression due to
the modified hypothesis to accomodate for the classification dependent variables.
So, the cost function is changed to the following

Cost(h_x,y) = 0.5(h_x - y)^2

But, the cost function still need modified, as the current cost funtion is still a non-convex
function. To accomidate for this, the cost function will now be

Cost(h_x,y) = 	-log(h_x) if y=1
		-log(1-h_x) if y=0


This works because we know that h_x can only be a value between 0 and 1,the -log()
function allows a more proper cost function evaluation.
For y=1 version, as h_x -> 1, -log(h_x) -> 0
For y=1 version, as h_x -> 0, -log(h_x) -> infinity

For y=0 version, as h_x -> 1, -log(1-h_x) -> infinity
For y=0 version, as h_x -> 0, -log(1-h_x) -> 0

Basically, this modified formula allows the cost function to properly reflect that
if the prediction (h_x) is far from the actual answer (y), then penalize the
hypothesis by the appropriate amount. Say if the prediction is 0.1, but the answer
is 1, then the cost would be exponentailly higher than if the prediction was 0.9
and the answer is 1.

To help simplify the cost function, we can write the function as

Cost(h_x,y) = -y*log(h_x) - (1-y)log(1-h_x)

This works because y can only be 1 or 0, so this one of the terms in the equation gets
cancelled out


GRADIENT DESCENT/MINIMIZATION OF THETAS


Theta[j] = Theta[j] - a(h_x - y)x
For all Theta[] simultaneously


a = learning rate
h_x = hypothesis
y = answer
x = current category (if j = 0, then x = 1)






Results
At an alpha rate of a=.005 and 1000 iterations, T=1000
theta = [-0.1519316074065917, -0.4190953198037469, 0.025087246039882286, 0.44698004075123826, 0.654407604796753, -0.38948107430466233, 0.642582119867666, 0.6090677339595952, 0.8638898130549236, -0.1977414715899427, -1.1041320435109705, 0.3849356705440782, 0.1643485857140098, 0.30590640360340954, 0.7383373907281978, 0.5194249407599058, 0.3765168258504252, 0.06601410228978351, 0.6143271629469965, 0.2951841333407749, 0.30042626644329545, 0.5007399562964415, -0.19449650646796426, 0.06122910996535835, 0.492635963796904, -0.7354184363853579, 0.3045690796727877, 0.8320975499768383, 0.4040348030802984, 0.11489770212115816, 0.2834245691546337]
This took ~45 minutes to create the model, and ended with an accuracy rate of ~87.521968% on the
training set data.
The cost started out at ~4.2 and after ~400 iteration cost
came down to ~0.5. At iteration 1000, cost came down to ~0.4


alpha rate = .0005 and 10,000 iterations
theta = [-0.13019641397227097, 0.5013903243371093, 0.15014756131861975, -0.10826523921500485, 0.08130600218304652, -0.29661766783335863, 0.6927028468698342, 0.8976642826559935, 0.7732330805110962, -0.6610710435969134, -0.6703958396248579, 0.576301472703391, 0.4914571911941294, 0.5833924703497689, 0.06947340525390948, 0.029887570007273767, 0.43752635923260774, 0.6024059034515972, -0.09959638870635232, 0.19253103921788406, 0.5122467190125878, -0.011269683036978702, -0.48080261639166927, 0.27948527274891577, 0.3807162814984631, -0.7232723696495073, 0.32821835829468715, 0.8669005234144801, 0.44344132634032746, 0.17144938508860846, 0.21617069309989176]
program began at 8pm ended at 9am
Acuracy was 85.237258%, making it less accurate than the previous









