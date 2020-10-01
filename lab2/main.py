import cv2
import numpy as np
from ehd import EHD
import csv
import sys

def main(argv):

    ehd = EHD(160, 120, 5)
    X_train = {}
    X_test = {}

    # read train dataset
    with open(argv[1], newline='') as csv_file:
        spamreader = csv.reader(csv_file, delimiter=' ', quotechar='|')
        for row in spamreader:
            X_train[row[0]] = int(row[1])

    # read test dataset
    with open(argv[2], newline='') as csv_file:
        spamreader = csv.reader(csv_file, delimiter=' ', quotechar='|')
        for row in spamreader:
            X_test[row[0]] = int(row[1])

    ehd.train(X_train)
    ehd.test(X_test)

    # write result train dataset
    with open(argv[3], 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|')
        spamwriter.writerow(['name', 'time_processing', 'shape'])
        for item in zip(ehd.X_train_name, ehd.time_train):
            img = cv2.imread(item[0], 0)
            spamwriter.writerow([item[0], item[1], img.shape])

    # write result test dataset
    with open(argv[4], 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|')
        spamwriter.writerow(['name', 'score', 'y_predict', 'y_test', 'time_processing', 'time_recognition', 'shape'])
        for item in zip(ehd.X_test_name, ehd.score, ehd.y_predict, ehd.y_test, ehd.time_test):
            img = cv2.imread(item[0], 0)
            spamwriter.writerow([item[0], item[1], item[2], item[3], item[4][0], item[4][1], img.shape])

    # print accurancy prediction
    print(np.sum((np.array(ehd.y_test) == np.array(ehd.y_predict))) / len(ehd.y_test))

if __name__ == '__main__':
    main(sys.argv)