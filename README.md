# Training the Model

## Setting up dependencies

For dowloading all the dependencies for training the yolov7 model execute the bash file named "dependencies.sh". This will dowload the model to current directory and all his dependencies in a virtual enviroment.


## Separating the Data.

For separating the dataset into differents subdatasets use the following two scripts

```
python k_fold.py --number 5 --dataset "./dataset" --outcsv "./data.csv"

```
This script will generate a `.csv` file that separate the data into n subfolders. The data will be randomly shuffle and equally distribute. This is made for using a cross validation testing of the model. In the case of just wanting to separate the data in the more general train, test and, val you can use the following script.

```
python separate.py --dataset "./dataset"
```

In the case of wanting to separate the folders with the cross validation technique you must specifie the following parameters

```
python separate --crossval --cvs "./data.csv" --dataset "./dataset" --training-session 1 
```

### Description of parameters:




The parameters of the  script are the following:

```
```

##Move the dataset to corresponding Folders:

Use the 


## Train the Model
For training the model you can click [here](https://github.com/WongKinYiu/yolov7) to see the github repo.If you already run the dependencies.sh it must have dowload the yolo7 repositorie so for training execute the following scripts.

```
cd yolov7

# train p5 models
python train.py --workers 8 --device 0 --batch-size 32 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml

# train p6 models
python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/coco.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6.yaml --weights '' --name yolov7-w6 --hyp data/hyp.scratch.p6.yaml
```
I will proceed to explain the code. first we call the python script train.py. This script take as parameter:
 - workers - The number of threads to use in the training.
  - devices - The id's of gpu's you want to use. In the case of only using a single gpu put --device 0
   - batch-size - The number of images that is gonna process for each training iteration.
 - img -In here you specified the  downsample dimensions for the image. When more big the ratio more computer power you will need. For basic training without the need of being to hacky just put --img 640 640.
  - cfg In here comes the configuration of the CNN model.  Don't mess with it if you don't know what parameters  you are are changing. For this you must have a very good intuition of what the model is really doing but for general cases just use the default parameters.

  - weights - This is for using pretained model weights in our training. You can put multiple weights path for this parameter.

  - name - When running the script it will create a `runs` folder. Inside it will create a folder with the name specified in this parameter. It will have the files and information of the project.

  - hyp - In here you must put the yaml file of the hyperparameters.All this values were already tested to best fit the performance on the [coco dataset](https://cocodataset.org/#home). Again if you don't know what are you modifying don't touch it.


## Testing
The model have a test.py script that gives us all the information of the model. To run it do this.
```
python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val
```
Parameters:
 - data - As the previous command, the data parameter is for input a yaml file that specifie the training, test and validation directory as well as the number of classes to identify and their names.
  - batch - Batch is the number of forward propagations we want to run at the same time. Because we are using Cuda we can benefit us from the parallelism of the GPU. 32 is normally the default value.

  -  conf - This stands for confident level.(TODO: Check what the heck this is for)

   - name - The name of the folder inside the `yolov7/runs/test/`.

   - weights - The path of the trained weights. Unlike the previous script, this param only admits one path becuase inference only use 1 weights file.

   - conf - This parameter is the confidence level of the object detected. This work as a threshold to not identify the objects that are less of the input number.

   - iou - This stands for `Intersection over Union`. The theory can become very difficult but to set things simple this is summary. There are two rects. The real one (The one that we input in the labeling) and the predicted one(The given one from the model). The iou is the ratio between the Union Area divided by the intersection area between the the rectangles. This parameter must be a float between 0 - 1 and filters all objects that this operation is less that the input float.
   


## Transfer Learning
Also called as Fine tuning, transfer learning is the way of using previous trained weights into our training. The advantage of this technique is to omit training of the  the model all over again each time we are training a new model. This take the weights given as a checkpoint and  start to optimize the parameters from there.

What precautions you must take if using transfer learning?
1. If you transfer a bad model to a new one, outcomes wont become better. Transfering .....


We are 


## Export the Model to onnx format
ONNX models are a generic format for exporting Neural Networks models. This is useful specially in our case because the Yolov7 model is based in PyTorch but our Vision module works on C++. What we do is to train the model in PyTorch and exported in a ONNX file so that the TensorRT inference  model ingest the onnx file and do the Inference from C++.
```
python export.py --weights yolov7-tiny.pt --grid --end2end --simplify \ --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640

```
## Inference
In the case of wanting to do inference directly from here you can easily use the following command.
