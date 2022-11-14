# Training the Model

## Setting up dependencies

For downloading all the dependencies for training the yolov7 model execute the bash file named "dependencies.sh". This will download the model to the current directory and all its dependencies in a virtual environment.


## Separating the Data.

For separating the dataset into different subdatasets use the following two scripts

```
python k_fold.py --number 5 --dataset "./dataset" --outcsv "./data.csv"

```
This script will generate a `.csv` file that separates the data into n subfolders. The data will be randomly shuffled and equally distributed. As an aclaration, in the `.csv` file won't be stored the supervised data(text file) of the image but you don't have to worry because when separating the images into their corresponding folder, the text file will be moved with it. This script is used in the case of using a cross-validation testing on the model. In the case of just wanting to separate the data in the more general train, test, and, val you can use the following script.

```
python separate.py --dataset "./dataset"
```

But if you want to separate the folders with the cross-validation technique, you must specify the following parameters.

```
python separate --crossval --cvs "./data.csv" --dataset "./dataset" --training-session 1 
```

### Description of parameters:
#### kfold.py 
- number - The number of datasets to generate in the .csv file. 
- dataset - The directory of the dataset.
- outcsv - The directory where the file will be saved. You must specify the name of the file inorder to work. Example:'./data.csv'.

#### separate.py
- crossval - This tag tells us if we want to apply a cross-validation separation on the dataset. If not, it will do a normal train(80%), test(10%) and, val(10%) separation.
 -  csv - The directory of the `.csv` file of the datasets information.
  - dataset - The directory of the dataset. This needs it because the .csv file only stores the names of the images but not where it is located in the machine.
  - training-session - The training iteration for the Cross-Validation test. The maximum number of training sessions will be the same as the number of datasets that we separate.  


## Train the Model
For training the model you can click [here](https://github.com/WongKinYiu/yolov7) to see the GitHub repo. If you already run the dependencies.sh it must have downloaded the yolo7 repository. For training execute the following scripts.

```
cd yolov7

# train p5 models
python train.py --workers 8 --device 0 --batch-size 32 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml

# train p6 models
python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/coco.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6.yaml --weights '' --name yolov7-w6 --hyp data/hyp.scratch.p6.yaml
```
I will proceed to explain the code. First, we call the python script train.py. This script takes as parameter:
 - workers - The number of threads to use in the training.
  - devices - The id of the GPU's you want to use. In the case of only using a single GPU put --device 0
   - batch-size - The number of images that is going to process for each training iteration.
 - img -In here you specified the downsample dimensions for the image. When more big the ratio more computer power you will need. For basic training without the need of being too hacky just put --img 640 640.
  - cfg In here comes the configuration of the CNN model.  Don't mess with it if you don't know what parameters you are changing. For this, you must have a very good intuition of what the model is doing but for general cases just use the default parameters.

  - weights - This is for using pre-trained model weights in our training. You can put multiple weights paths for this parameter.

  - name - When running the script it will create a `runs` folder. Inside, it will create a folder with the name specified in this parameter. It will have the files and information on the project.

  - hyp - In here you must put the YAML file of the hyperparameters. All these values were already tested to best fit the performance on the [coco dataset](https://cocodataset.org/#home). Again if you don't know what are you modifying don't touch it.


## Testing
The model has a test.py script that gives us all the information of the model. To run it do this.
```
python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val
```
Parameters:
 - data - As in the previous command, the data parameter is for inputting a YAML file that specified the training, test, and validation directory as well as the number of classes to identify and their names.
  - batch - Batch is the number of forward propagations we want to run at the same time. Because we are using Cuda we can benefit from the parallelism of the GPU. 32 is normally the default value.

  -  conf - This stands for confidence level. This parameter is a filter to only output the detections whose confidence levels are equal to or above this threshold.

   - name - The name of the folder inside the `yolov7/runs/test/`. directory.

   - weights - The path of the trained weights. Unlike the previous script, this param only admits one path because the inference can only use 1 weights file.

   - conf - This parameter is the confidence level of the object detected. This work as a threshold to not identify the objects that are less of the input number.

   - iou - This stands for `Intersection over Union. The theory can become very difficult but to set things simply. There are two rects. The real one (The one that we input in the labeling) and the predicted one(The one given from the model). The iou is the ratio between the Union Area divided by the intersection area between the rectangles. This parameter must be a float between 0 - 1 and filter all objects that this operation is less than the input float. This is basically how we measure how good was the predicted bounding box.
   


## Transfer Learning
Also called Fine-tuning, transfer learning is the way of using previously trained weights into our training. The advantage of this technique is to omit the training all over again each time we are training a new model. This takes the weights given as a checkpoint and starts to optimize the parameters from there.

What precautions you must take if using transfer learning?
If you transfer a bad model to a new one, outcomes won't become better. If the given weights are already biased the new model will learn the same biased information.

Besides that fine-tuning has proved to be very beneficial in a lot of applications and has very good general results. In the case of our object detection problem is very recommended. We can use the yolov7.pt weights that have been trained on the Coco dataset and for a small dataset like ours the performance can improve a lot.




We are 


## Export the Model to onnx format
ONNX models are a generic format for exporting Neural Networks models. This is useful especially in our case because the Yolov7 model is based in PyTorch but our Vision module works on C++. So What we do is train the model in PyTorch and exported it in an ONNX file so that the TensorRT inference model ingests the onnx file and does the Inference from C++.
```
python export.py --weights yolov7-tiny.pt --grid --end2end --simplify \ --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640

```
## Inference
In the case of wanting to do inference directly from Python, you can easily use the following command.
```bash
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source yourvideo.mp4
```
or 
```bash
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
```