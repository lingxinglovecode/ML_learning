from matplotlib import pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from SVM.MNIST_Digit_recognition.main import My_SVM
data_dir = 'D://personal_file//Github//ML_learning//data'
lfw_prople = fetch_lfw_people(data_home=data_dir,min_faces_per_person=70, resize=0.4)
a = 2
n_samples,h,w = lfw_prople.images.shape

X = lfw_prople.data
Y = lfw_prople.target
target_names = lfw_prople.target_names
n_classes = target_names.shape[0]
#split train and test
X_train, X_test, y_train, y_test =  train_test_split(X,Y,test_size=0.25,shuffle=True,random_state=1)

n_components = 150
#pca to extract features
pca = PCA(n_components=n_components,whiten=True).fit(X_train)

#X_train_pca = pca.transform(X_train)
#X_test_pca = pca.transform(X_test)
#eigenfaces = pca.components_.reshape((n_components, h, w))

X_train_pca = X_train
X_test_pca = X_test
#train a SVM classification model

svm_model = SVC(kernel='rbf')
svm_model.fit(X_train_pca,y_train)
test_score = svm_model.score(X_test_pca,y_test)
print(test_score)

print("Fitting the classifier to the training set")
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced'), param_grid
)
clf = clf.fit(X_train_pca, y_train)
print("Best estimator found by grid search:")
print(clf.best_estimator_)

 #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))



# #############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()
