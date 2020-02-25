import cv2
import numpy as np
import os
import shutil
from six.moves import cPickle as pickle #for performance
import wget
import time
import tarfile
from itertools import compress
from helpers import *
from matplotlib import pyplot as plt


class BOV:
    def __init__(self, no_clusters):
        self.no_clusters = no_clusters
        self.train_path = None
        self.test_path = None
        self.im_helper = ImageHelpers()
        self.bov_helper = BOVHelpers(no_clusters)
        self.file_helper = FileHelpers()
        self.images = None
        self.trainImageCount = 0
        self.train_labels = np.array([])
        self.name_dict = {}
        self.descriptor_list = []

    def save_obj(self, di_, filename_):
        with open(filename_, 'wb') as f:
            pickle.dump(di_, f)

    def load_obj(self, filename_):
        with open(filename_, 'rb') as f:
            ret_di = pickle.load(f)
        return ret_di

    def trainModel(self):
        """
        This method contains the entire module
        required for training the bag of visual words model
        Use of helper functions will be extensive.
        """

        if not os.path.exists("descriptores.pkl"):
            # read file. prepare file lists.
            self.images, self.trainImageCount = self.file_helper.getFiles(self.train_path)
            # extract SIFT Features from each image
            label_count = 0
            cuenta = 0
            for word, imlist in self.images.items():
                self.name_dict[str(label_count)] = word
                print ("Computing Features for ", word)
                for im in imlist:
                    # cv2.imshow("im", im)
                    # cv2.waitKey()
                    self.train_labels = np.append(self.train_labels, label_count)
                    kp, des = self.im_helper.features(im)
                    self.descriptor_list.append(des)

                    # uncoment these lines to watch and save images with their keypoints.
                    #img=cv2.drawKeypoints(im,kp, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    #cv2.imwrite(f'Resultados/sift_keypoints{cuenta}.jpg',img)
                    #cuenta = cuenta + 1
                    #cv2.imshow("imagen", img)
                    #cv2.waitKey(1)

                label_count += 1

            self.save_obj(self.descriptor_list, "descriptores.pkl")
            np.save("anotaciones", self.train_labels, allow_pickle=True)
            self.save_obj(self.name_dict, 'nombres_clases.pkl')
            CuentaImagenesTrain = np.asarray([self.trainImageCount])
            np.save("numeroImagenes", CuentaImagenesTrain, allow_pickle=True)

        else:
            self.descriptor_list = self.load_obj("descriptores.pkl")
            self.train_labels = np.load("anotaciones.npy")
            self.name_dict = self.load_obj('nombres_clases.pkl')
            self.trainImageCount = np.load("numeroImagenes.npy")[0]


        # perform clustering
        bov_descriptor_stack = self.bov_helper.formatND(self.descriptor_list)
        t3 = time.time()
        self.bov_helper.cluster()
        t4 = time.time()
        tiempokmean = t4-t3
        self.bov_helper.developVocabulary(n_images = self.trainImageCount, descriptor_list=self.descriptor_list)

        # show vocabulary trained
        self.bov_helper.plotHist()


        self.bov_helper.standardize()
        self.bov_helper.train(self.train_labels)
        return tiempokmean


    def recognize(self,test_img, test_image_path=None):

        """
        This method recognizes a single image
        It can be utilized individually as well.
        """

        kp, des = self.im_helper.features(test_img)
        # print kp
        #print( des.shape)

        # generate vocab for test image
        vocab = np.array( [[ 0 for i in range(self.no_clusters)]])
        # locate nearest clusters for each of
        # the visual word (feature) present in the image

        # test_ret =<> return of kmeans nearest clusters for N features
        test_ret = self.bov_helper.kmeans_obj.predict(des)
        # print test_ret

        # print vocab
        for each in test_ret:
            vocab[0][each] += 1

        #print (vocab)
        # Scale the features
        vocab = self.bov_helper.scale.transform(vocab)

        # predict the class of the image
        lb = self.bov_helper.clf.predict(vocab)
        # print "Image belongs to class : ", self.name_dict[str(int(lb[0]))]
        return lb



    def testModel(self):
        """
        This method is to test the trained classifier
        read all images from testing path
        use BOVHelpers.predict() function to obtain classes of each image
        """

        self.testImages, self.testImageCount = self.file_helper.getFiles(self.test_path)

        predictions = []

        for word, imlist in self.testImages.items():
            #print ("processing " ,word)
            for im in imlist:
                # print imlist[0].shape, imlist[1].shape
                #print (im.shape)
                cl = self.recognize(im)
                #print (cl)
                predictions.append({
                    'image':im,
                    'class':cl,
                    'object_name':self.name_dict[str(int(cl[0]))],
                    'real_name':word
                    })

        #print (predictions)
        #for each in predictions:
            # cv2.imshow(each['object_name'], each['image'])
            # cv2.waitKey()
            # cv2.destroyWindow(each['object_name'])
            #
            #plt.imshow(cv2.cvtColor(each['image'], cv2.COLOR_GRAY2RGB))
            #plt.title(each['object_name'])
            #plt.show()
        return predictions


    def print_vars(self):
        pass

def check_dataset(folder):

    if not os.path.isdir(folder):
        url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz'
        filename = wget.download(url)
        tar = tarfile.open(filename, "r:gz")
        tar.extractall()
        tar.close()
    else:
        print("directorio ya creado")

    if not os.path.isdir(os.path.join(folder, "train")):
        splitDatabase(folder);
    else:
        print("Base de datos ya separada en train y test")

def splitDatabase(folder2):

    categories = os.listdir(folder2)
    # Create target directory & all intermediate directories if don't exists
    dirName = os.path.join(folder2, "train")
    try:
        os.makedirs(dirName)
        print("Directory " , dirName ,  " Created ")
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")

    for cat in categories:
        origin_path = os.path.join(folder2, cat)
        dest_dir = os.path.join(folder2, "train")
        if not origin_path == os.path.join(folder2, "BACKGROUND_Google"):
            shutil.move(origin_path, dest_dir)

    base_path = os.getcwd()
    data_path = os.path.join(base_path, "101_ObjectCategories/train")
    categories = os.listdir(data_path)
    test_path = os.path.join(base_path, "101_ObjectCategories/val")
    if not os.path.isdir(os.path.join(folder2, "val")):
        dirName = os.path.join(folder2, "val")
        try:
            os.makedirs(dirName)
            print("Directory " , dirName ,  " Created ")
        except FileExistsError:
            print("Directory " , dirName ,  " already exists")

    for cat in categories:
        image_files = os.listdir(os.path.join(data_path, cat))
        choices = np.random.choice([0, 1], size=(len(image_files),), p=[.80, .20])
        files_to_move = compress(image_files, choices)

        for _f in files_to_move:
            origin_path = os.path.join(data_path, cat,  _f)
            dest_dir = os.path.join(test_path, cat)
            dest_path = os.path.join(test_path, cat, _f)
            if not os.path.isdir(dest_dir):
                os.mkdir(dest_dir)

            shutil.move(origin_path, dest_path)

if __name__ == '__main__':

    t0 = time.time()
    path_DB = "101_ObjectCategories/train"

    check_dataset(path_DB.split('/')[0])
    #print(path_DB.split('/')[0])

    bov = BOV(no_clusters=200)

    # set training paths
    bov.train_path = "101_ObjectCategories/train"
    # set testing paths
    bov.test_path = "101_ObjectCategories/val"
    # train the model
    tiempoKMeans = bov.trainModel()
    print('tiempo del k-means: ')
    print(tiempoKMeans)
    # test model
    pred = bov.testModel()

    cm, ACA = bov.bov_helper.evaluacion(pred, np.array(list(bov.name_dict.values())))

    t1 = time.time()
    total = t1-t0
    print('tiempo total: ')
    print(total)
    #print(cm)
    print('ACA= ')
    print(ACA)
    print('ACA_train= ')
    print( bov.bov_helper.clf.score(bov.bov_helper.mega_histogram, bov.train_labels))
