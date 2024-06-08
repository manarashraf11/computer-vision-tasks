import glob
import math
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
import matplotlib.image as mpimg

#getting the images paths
def get_filepaths(directory):
    
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(filename)
            file_paths.append(filepath)
#     print(file_paths)
#     print(len(file_paths))  # Add it to the list.

    return file_paths  # Self-explanatory.

#to know the number of classes(faces)
def class_calc(images_paths,training_path):
    
    categories=[]
    for image_path in images_paths:
        category=(image_path.split(".")[0]).split("_")[1]
        if int(category) not in categories:
            categories.append(int(category))
    categories= sorted(categories)
    class_num=len(categories)
    print(categories)
    return categories,class_num
#to get the training images and flattened array

def training(training_path):
    images_paths= get_filepaths(training_path)#get paths
    height = 640
    width = 480
    training_images = np.ndarray(shape=(len(images_paths), height*width), dtype=np.float64)
     #read the images and get the 1D vector
    for i in range(len(images_paths)):
        path= training_path+'/'+ images_paths[i]
        read_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(read_image, (width, height))
    #     print(resized_image.shape)
        training_images[i,:] = np.array(resized_image, dtype='float64').flatten()
        
        
    #     plt.imshow(resized_image, cmap='gray')
    #     plt.show()
    print(training_images.shape)
    return training_images

#to get the mean face 
def get_mean_normalized(training_path,training_images):
    images_paths= get_filepaths(training_path)#training images
    height = 640
    width = 480

    ##Get Mean Face##
    mean_face = np.zeros((1,height*width))

    for i in training_images:
        mean_face = np.add(mean_face,i)#to sum all images

    mean_face = np.divide(mean_face,float(len(images_paths))).flatten()# to get the mean face by dividing the summation by the length of images
    ##Normailze Faces##
    normalised_training = np.ndarray(shape=(len(images_paths), height*width))

    for i in range(len(images_paths)):
        normalised_training[i] = np.subtract(training_images[i],mean_face)#to substract mean face from each image

    return mean_face,normalised_training

#get the covariance matrix
def cov_mat(normalised_training):
    
    cov_matrix = ((normalised_training).dot(normalised_training.T))#dot product
    cov_matrix = np.divide(cov_matrix ,float(len(normalised_training)))
    return cov_matrix

#to get eigen faces and eigen vectors
def eigen_val_vec(cov_matrix):
    eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)
    eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]
    # Sort the eigen pairs in descending order:
    eig_pairs.sort(reverse=True)
    eigvalues_sort  = [eig_pairs[index][0] for index in range(len(eigenvalues))]
    eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]
    eigenfaces = preprocessing.normalize(eigvectors_sort)#normalize eigen vectors
    # print(eigenfaces.shape)
    return eigvalues_sort, eigenfaces
#to get the eigen faces till 90%
def get_reduced(eigenfaces,eigvalues_sort):
    var_comp_sum = np.cumsum(eigvalues_sort)/sum(eigvalues_sort)

    # Show cumulative proportion of varaince with respect to components
    # print("Cumulative proportion of variance explained vector: \n%s" %var_comp_sum)

    # x-axis for number of principal components kept
    # num_comp = range(1,len(eigvalues_sort)+1)
    # plt.title('Cum. Prop. Variance Explain and Components Kept')
    # plt.xlabel('Principal Components')
    # plt.ylabel('Cum. Prop. Variance Expalined')

    # plt.scatter(num_comp, var_comp_sum)
    # plt.show()

    #eigen faces components
    reduced_data=[]
    for i in (var_comp_sum):
        if i < 5:
            reduced_data.append(i)
    # print(reduced_data)
    # print(len(reduced_data))
    reduced_data = np.array(eigenfaces[:len(reduced_data)]).transpose()
    return reduced_data

#data projection
def projected_data(training_images,reduced_data):
    proj_data = np.dot(training_images.transpose(),reduced_data)
    proj_data = proj_data.transpose()
    print(proj_data.shape)
    return proj_data

#get weights of images
def weights(proj_data,normalised_training):
    w = np.array([np.dot(proj_data,i) for i in normalised_training])
    return w

#apply pca
def pca(unknown_face):

    training_path="our_faces/data"
    height=640
    width=480

    # unknown_face = cv2.imread(path_unknown, cv2.IMREAD_GRAYSCALE)#read the image
    gray = cv2.cvtColor(unknown_face, cv2.COLOR_BGR2GRAY)

    unknown_face = cv2.resize(gray, (width, height))  #resize with 80*70 shape
    # print(unknown_face.shape)
    unknown_face_vector = np.array(unknown_face, dtype='float64').flatten() #get the flattened array
    training_images=training(training_path)#
    mean_face,normalised_training=get_mean_normalized(training_path,training_images)
    print('--------------')

    normalised_uface_vector = np.subtract(unknown_face_vector,mean_face)
    print('00000000000')
    cov_matrix=cov_mat(normalised_training)
    eigvalues_sort, eigenfaces=eigen_val_vec(cov_matrix)
    print('111111111111')

    reduced_data=get_reduced(eigenfaces,eigvalues_sort)
    proj_data=projected_data(training_images,reduced_data)
    print('333333333333333333')

    w=weights(proj_data,normalised_training)
    w_unknown = np.dot(proj_data, normalised_uface_vector)#get the weight of test face
    euclidean_distance = np.linalg.norm(w - w_unknown, axis=1)#get the euclidean distance
    best_match = np.argmin(euclidean_distance)#get the index of the best matched one
    
    output_image= training_images[best_match].reshape(640,480)
    saved=mpimg.imsave('our_faces/FaceRecognized.png', output_image,cmap="gray")
    # Visualize

    # fig, axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,6))
    # axes[0].imshow(unknown_face_vector.reshape(80,70), cmap="gray")
    # axes[0].set_title("Query")
    # axes[1].imshow(training_images[best_match].reshape(80,70), cmap="gray")
    # axes[1].set_title("Best match")
    # plt.show()

# pca("data/test/149_15.jpg")




        