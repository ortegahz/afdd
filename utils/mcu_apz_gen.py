import numpy as np

from utils import svm_label2data_v2

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# test_data_gen = ImageDataGenerator(rescale=1. / 255)
# test_generator = test_data_gen.flow_from_directory('/home/manu/tmp/6/',
#                                                    target_size=(224, 224),
#                                                    # batch_size=nb_test_files,
#                                                    class_mode='categorical')
# test_data = next(test_generator)
# x_test, y_test = test_data
#
# # or to have a simple file
# np.savez("/home/manu/tmp/mydata.npz", x_test=x_test, y_test=y_test)

x_test, y_test = svm_label2data_v2("/home/manu/tmp/afd_pm_train", 16)

print(x_test)
print(y_test)

# np.savez("/home/manu/tmp/mydata.npz", x_test=x_test, y_test=y_test)

np.savez("/home/manu/tmp/x_test.npz", x_test=x_test)
np.savez("/home/manu/tmp/y_test.npz", y_test=y_test)