Code Summary:
Load Data: The Iris dataset is loaded, using only the first two features for easy visualization.

Normalize Features: Features are scaled using StandardScaler so that distance calculations are fair.

Train/Test Split: Data is split into training and testing sets.

Train KNN: For different values of K (1, 3, 5, 7, 9), a KNeighborsClassifier is trained and evaluated.

Evaluate: The accuracy and confusion matrix are displayed for each K.

Visualize: A decision boundary is plotted to show how KNN classifies different regions of the feature space.
