
def getNamesDirectory(path):
    list_img=[img for img in os.listdir(path) if img.endswith('.png')==True]

    path_img=[]

    for i in range (len(list_img)):
        path_img.append(list_img[i])
        
    directories = np.array(path_img)
    return directories


# list_directory parameter must be a numpy array inorder to not give an error.
# This is because we are selecting multiple indexes at once in the array,
#  something that python list's can't do.
def createCSV(list_directory, split_number,output_dir):
    
    groups = np.zeros(len(list_directory))
    group_kfold = KFold(n_splits=split_number,shuffle=True) 
    dict_to_csv = {}
    counter = 0
    list_max = 0
    for train_index, test_index in group_kfold.split(list_directory, groups = groups):
        counter += 1
        list_len = len(list_directory[test_index])

        dict_to_csv[counter] = list(list_directory[test_index])

        if list_max < list_len:
            list_max = list_len

    #This step is to fill with blank strings the arrays so it can be rectangular 
    # and it can be converted to a *.csv file
    for dataset in dict_to_csv:
        for i in range(list_max - len(dict_to_csv[dataset])):
            dict_to_csv[dataset].append("") 

    df = pd.DataFrame(dict_to_csv)
    df.to_csv(output_dir)

    



if __name__ == "__main__":
    import os
    import pandas as pd
    from sklearn.model_selection import KFold
    import numpy as np
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", help = "Specified the number of foldings", required= True)
    parser.add_argument("--dataset", help = "directory of the dataset", required= True)
    parser.add_argument("--outcsv", help = "output directory of the csv", required= True)
    args = parser.parse_args()

    directories = getNamesDirectory(args.dataset)
    createCSV(directories, int(args.number), args.outcsv)