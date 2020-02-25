import cv2
import numpy as np
from glob import glob
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

class ImageHelpers:
    def __init__(self):
        ##  WARNING !!
        ##
        ##      If parameter nfeatures in SIFT is change, you have to create new files to store the descriptor list
        ##      please after change it delete files .pkl and .npy in your root directory
        ##
        ##  WARNING !!
        self.sift_object = cv2.xfeatures2d.SIFT_create(nfeatures = 300)

    def gray(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def features(self, image):
        keypoints, descriptors = self.sift_object.detectAndCompute(image, None)
        return [keypoints, descriptors]


class BOVHelpers:
    def __init__(self, n_clusters = 20):
        self.n_clusters = n_clusters
        self.kmeans_obj = MiniBatchKMeans(n_clusters = n_clusters, batch_size=100, random_state=123 )
        self.kmeans_ret = None
        self.descriptor_vstack = None
        self.mega_histogram = None
        self.clf  = SVC(kernel=chi2_kernel)

    def my_kernel(self, X, Y):
        """
        We create a custom kernel:

        k(X, Y) = min(X,Y)
        """
        m_m = np.asarray([np.minimum(x, Y).sum(axis = 1) for x in X])
        return m_m

    def evaluacion(self, predicciones, lab):

        tabla = pd.DataFrame(predicciones)
        y_true = np.asarray(tabla['real_name'])
        y_pred = np.asarray(tabla['object_name'])
        h = metrics.confusion_matrix(y_true,y_pred, labels=lab)
        ACA = metrics.accuracy_score(y_true, y_pred)

        return h, ACA

    def cluster(self):
        """
        cluster using KMeans algorithm,
        """
        print('inicio de k-means')
        self.kmeans_ret = self.kmeans_obj.fit_predict(self.descriptor_vstack)

    def developVocabulary(self,n_images, descriptor_list, kmeans_ret = None):

        """
        Each cluster denotes a particular visual word
        Every image can be represeted as a combination of multiple
        visual words. The best method is to generate a sparse histogram
        that contains the frequency of occurence of each visual word
        Thus the vocabulary comprises of a set of histograms of encompassing
        all descriptions for all images
        """

        self.mega_histogram = np.array([np.zeros(self.n_clusters) for i in range(n_images)])
        old_count = 0
        for i in range(n_images):
            l = len(descriptor_list[i])
            for j in range(l):
                if kmeans_ret is None:
                    idx = self.kmeans_ret[old_count+j]
                else:
                    idx = kmeans_ret[old_count+j]
                self.mega_histogram[i][idx] += 1
            old_count += l
        print ("Vocabulary Histogram Generated")

    def standardize(self, std=None):
        """

        standardize is required to normalize the distribution
        wrt sample size and features. If not normalized, the classifier may become
        biased due to steep variances.
        """
        if std is None:
            self.scale = Normalizer(norm='l1').fit(self.mega_histogram)
            self.mega_histogram = self.scale.transform(self.mega_histogram)
        else:
            print ("STD not none. External STD supplied")
            self.mega_histogram = std.transform(self.mega_histogram)

    def formatND(self, l):
        """
        restructures list into vstack array of shape
        M samples x N features for sklearn
        """
        print(len(l))
        vStack = np.vstack(l)
        self.descriptor_vstack = vStack.copy()
        return vStack

    def train(self, train_labels):
        """
        uses sklearn.svm.SVC classifier (SVM)
        """
        print ("Training SVM")
        print (self.clf)
        print ("Train labels", train_labels)
        self.clf.fit(self.mega_histogram, train_labels)
        print ("Training completed")

    def predict(self, iplist):
        predictions = self.clf.predict(iplist)
        return predictions

    def plotHist(self, vocabulary = None):
        print( "Plotting histogram")
        if vocabulary is None:
            vocabulary = self.mega_histogram

        x_scalar = np.arange(self.n_clusters)
        y_scalar = np.array([abs(np.sum(vocabulary[:,h], dtype=np.int32)) for h in range(self.n_clusters)])

        print( y_scalar)

        plt.bar(x_scalar, y_scalar)
        plt.xlabel("Visual Word Index")
        plt.ylabel("Frequency")
        plt.title("Complete Vocabulary Generated")
        plt.xticks(x_scalar + 0.4, x_scalar)
        plt.savefig('Resultados/hist/histograma_total.png')
        plt.show()


class FileHelpers:
    def __init__(self):
        pass

    def getFiles(self, path):
        """
        - returns  a dictionary of all files
        having key => value as  objectname => image path
        - returns total number of files.
        """
        imlist = {}
        count = 0
        for each in glob(path + "/*"):
            word = each.split("\\")[-1]
            print (" #### Reading image category ", word, " ##### ")
            imlist[word] = []
            for imagefile in glob(path+"/"+word+"/*"):
                print ("Reading file ", imagefile)
                im = cv2.imread(imagefile, 0)
                imlist[word].append(im)
                count +=1

        return [imlist, count]
