# Todo: 
# (1) Write the Function to separate the Folders Implementing tqdm.
# (2) Write the Function to execute the training of the training session
# (3) Write the documentation and all the steps to do the training and testing in a read.me
# (4) make a bash file for dowloading all the dependencies.
# (5) Make the GitHub repositorie

def getNamesDirectory(path):
    list_img=[img for img in os.listdir(path) if img.endswith('.png')==True]

    path_img=[]

    for i in range (len(list_img)):
        path_img.append(list_img[i])
        
    directories = np.array(path_img)
    return directories


def separateToCross(train_session, csvdir, dataset_dir = "./dataset",):
    #  Check if train and test folders exist in current folder
    istrain = os.path.exists("./train")
    if istrain == False:
        os.system("mkdir train")
    istest = os.path.exists("./test")
    if istest ==  False:
        os.system("mkdir test")
    isval = os.path.exists("./val")
    if isval == False:
        os.system("mkdir val")

    #  If train test and val folders exist, delete the content of them
    if istrain == True:
        os.system("rm -r train/*")
    if istest == True:
        os.system("rm -r test/*")
    if isval == True:
        os.system("rm -r val/*")


    #  Read from a csv file and separate the images into  the train, test and val folders
    df = pd.read_csv(csvdir)
    for col in df.columns:
        print(col)
        print(list(df.columns))

    columns = list(df.columns)[1:]
    columns.remove(train_session)    
    train_pd = pd.concat([df[i] for i in columns]).tolist()
    test_pd = df[train_session].tolist()

    #  Separate randomly the elements in test_pd into test and val
    test_pd, val_pd = train_test_split(test_pd, test_size=0.5, random_state=42)

    #  Move the images to the corresponding folders
    for i in tqdm(range(len(train_pd))):
        os.system("cp " + dataset_dir + "/" + train_pd[i] + " train/" )
        os.system("cp " +dataset_dir+ "/" + train_pd[i].replace(".png", ".txt") + " ./train")

    for i in tqdm(range(len(test_pd))):
        os.system("cp " +dataset_dir+ "/" +test_pd[i] + " ./test")
        os.system("cp " +dataset_dir+ "/" +test_pd[i].replace(".png", ".txt") + " ./test")
    
    for i in tqdm(range(len(val_pd))):
        os.system("cp " +dataset_dir+ "/" + val_pd[i] + " ./val")
        os.system("cp " +dataset_dir+ "/" + val_pd[i].replace(".png", ".txt") + " ./val")

def separateData(dataset_dir = "./dataset"):
    istrain = os.path.exists("./train")
    if istrain == False:
        os.system("mkdir train")
    istest = os.path.exists("./test")
    if istest ==  False:
        os.system("mkdir test")
    isval = os.path.exists("./val")
    if isval == False:
        os.system("mkdir val")

    #  If train test and val folders exist, delete the content of them
    if istrain == True:
        os.system("rm -r train/*")
    if istest == True:
        os.system("rm -r test/*")
    if isval == True:
        os.system("rm -r val/*")

    dataset = getNamesDirectory(dataset_dir)
    #  Separate randomly the elements of the dataset into train, test and val
    train, test = train_test_split(dataset, test_size=0.1, shuffle=True)
    train, val = train_test_split(train, test_size=0.11, shuffle=True)

    #  Move the images to the corresponding folders
    for i in tqdm(range(len(train_pd))):
        os.system("cp " + dataset_dir + "/" + train[i] + " train/" )
        os.system("cp " +dataset_dir+ "/" + train[i].replace(".png", ".txt") + " ./train")

    for i in tqdm(range(len(test_pd))):
        os.system("cp " +dataset_dir+ "/" +test[i] + " ./test")
        os.system("cp " +dataset_dir+ "/" +test[i].replace(".png", ".txt") + " ./test")
    
    for i in tqdm(range(len(val_pd))):
        os.system("cp " +dataset_dir+ "/" + val[i] + " ./val")
        os.system("cp " +dataset_dir+ "/" + val[i].replace(".png", ".txt") + " ./val")


if __name__ == "__main__":
    import argparse
    import tqdm 
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-session",help= "Specifie what training session we are doing in the cross validation")
    parser.add_argument("--csv", help= "the csv file that have the shuffle training dataset")
    parser.add_argument("--dataset", help= "the directory of the dataset")
    parser.add_argument("--crossval", help= "the directory of the train folder")
    # add a argument to specified to only separate the folders
    parser.parse_args()
    args = parser.parse_args()

    if args.crossval:
        separateToCross(args.training_session, args.csv, args.dataset)
    else:
        separateData(args.dataset)
        