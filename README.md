# PredictingMaskAndDepthUsingDeepLearning
Predicting Mask and Depth of image from background and foreground superimposed on background Using Deep Learning Techinques

# I. Problem Statement

In this project I have used Neural Transformer to generate Python Source Code from a given English Description. 

I have used a custom dataset where each record starts with English Description line starting with # character followed 
by python code.

The dataset is available in link below:

https://drive.google.com/file/d/116ClZ6nu1kL-RUCx-p3Sc-SGIq1zB9a-/view?usp=sharing

Jupyter Notebook link:


Youtube Link:

Major Highlights:

-  Use fully API based implementation 
-  Used Neural Transformer based architecture
-  In decoder used token type embedding in addition to token embedding and positional embedding
-  Built custom Python Tokenizer to tokenize python code and type
-  Used spacy English Tokenizer to tokenize english description
-  Generated Glove Embeddings for Python Code and English separately from the scratch and used these embeddings during training
-  Used a composite cross entry loss function which combines python token loss and python token type loss
-  Deployed the model in AWS EC2 and built a website using Flask


- Only 5 epochs are used to achive the result

## Running the code

For Training, you can use download the code

git clone https://github.com/monimoyd/PredictingMaskAndDepthUsingDeepLearning.git 

Alternatively , if you directly want to use the Jupyter Notebook, please take API from the google drive link:

https://drive.google.com/drive/folders/1YTvb7V0eDfn5MZwBbc4msFkWKH5ArotI?usp=sharing 


## II. Data Cleaning and Data Loading

The original custom python dataset was very messy and cluttered and it can not be processed by programs. 
Original dataset is changed manually to come up with the following format

1. English Description starts in a new line starting with # (Hash character)
2. Python Code will be immediately afte the English Description.
3. Each example pair is separated by a single blank line
4. There should not be any blank line in python code

Even after manual changes, faced some issues, written a custom parser to find the lines where the problem happens

Written a loader for loading the custom Python Dataset. The following cleanups are done

i.  Any Comments in Python Code are removed
ii. Removed all the import statements in python code in dataset as there is a pyforest package available which saves the effort of generating any import statements after the package is installed.
any import statement. While generating python code just need to include "import pyforest" statement
iii. From the english description all the punctulations are removed and all the words are made lower case 
  
Use json loader for loading Conala dataset for 

Used Pytext and BucketIterator for generating datasets

## i. Data Loader

Datal Loader performs loading of data from the images.


The workflow for dataloader is explained in the flowchart below:

![Project Report](/doc_images/data_loader_workflow.png)

The process involved:
, 
- Copy all the zip files from the google drive to the Google colab local folder /content/data
- Unzip each of zip in a respective batch folder. For example batch1_images.zip is unzipped to /content/data/batch1 folder.

Similar process is done for other batches as well
- There are two datasets:
  i. TrainImageDataset - This dataset is constructed from 9 zip files (batch1_images.zip, batch1_images.zip, ... batch9_images.zip) unzipped in
respetive batch folder (batch1, batch2 4, .. batch9)  and used for training.
  ii. TestImageDataset - This dataset is constructed using only batch10_images.zip unzipped in batch10 folder

Records are populated as below
 - Multi level index (batch id, offset) of all the files in fg_bg_jpg folder
  - The  __getitem__ method takes index as an argument.
 - index is used to calculate batch_id by dividing index by 40000. Remainder is used to calculate offset
 - Once the fg_bg image file is identified the corresponding background image file is identified based on naming convention.
 For exmaple of fg_bg image file name is fg_bg_1_100_1_15.jpg then by convention second number after fg_bg will be background
 image, in this case it will be bg_100.jpg and it will be avaialble in bg_jpg folder under respective batch id directory
 
 Based on convention the ground truth mask image, filename will have same suffix as the fg_bg image file name. For example if fg_bg image
 filename is fg_bg_1_100_1_15.jpg, file name correspoding to ground truth mask image will be 
 bg_mask_1_100_1_15.jpg, which will be available in mask_black_jpg folder under the batch id directory
 
 Similary, ground truth  depth image filename will have the same suffix as the fg_bg image file name. For example, if fg_bg image
 filename is fg_bg_1_100_1_15.jpg, the  filename correspoding depth image will be 
 depth_1_100_1_15.jpg, which will be available in depth_fg_bg_jpg directory under the respective batch directory
 
 The code for dataloader is available in URL:
 
 https://github.com/monimoyd/PredictingMaskAndDepthUsingDeepLearning/blob/master/data_loaders/fg_bg_images_data_loader.py
  

## II. Data Augmentation

### A. Adding conala dataset

For the training I have used combined the custom dataset with publicly avaialble conala dataset

https://conala-corpus.github.io/

This dataset is crawled from Stack Overflow, automatically filtered, then curated by annotators, split into 2,379 training and 500 test examples.

Conala dataset is in josn json format, one example is as below:

{
  "question_id": 36875258,
  "intent": "copying one file's contents to another in python", 
  "rewritten_intent": "copy the content of file 'file.txt' to file 'file2.txt'", 
  "snippet": "shutil.copy('file.txt', 'file2.txt')", 
}

I have combined all the conala datasets (2,379 training and 500 test examples.) and from two examples from each example, one based 
"intent" and another using "rewritten_intent". so total of 2*(2379 + 500) = 5758 example I have combined into the training dataset

The conala combined dataset is available below:

https://drive.google.com/file/d/1QO7TS2Oh2Vx-vcPVgeIolAr3QkQTM0Ua/view?usp=sharing

In addition, I have used the following text augmentation techniques 


### B. Synonym Replacement

First, you could replace words in the sentence with synonyms, like so:

The dog slept on the mat

could become

The dog slept on the rug

Aside from the dog's insistence that a rug is much softer than a mat, the meaning of the sentence hasn’t changed. But mat
and rug will be mapped to different indices in the vocabulary, so the model will learn that the two sentences map to the
same label, and hopefully that there’s a connection between those two words, as everything else in the sentences is the same
 

### C. Random Swap

The random swap augmentation takes a sentence and then swaps words within it n times, with each iteration working on the
previously swapped sentence. Here we sample two random numbers based on the length of the sentence, and then just keep
swapping until we hit n.
 

### D. Random Deletion

As the name suggests, random deletion deletes words from a sentence. Given a probability parameter p, it will go through the
sentence and decide whether to delete a word or not based on that random probability. Consider of it as pixel dropouts 
while treating images. 
 
 
### E. Back Translation

Back Translation involves translating a sentence from our target language into one or more other languages and then translating all of them back to the original language. 
I have  used google_trans_new Python library for this purpose. Note that google_trans_new Python library is very slow, so I used
only 5 percent of my total training set for performing Back Translation

## iii. Glove Embedding

{Source: https://nlp.stanford.edu/projects/glove/]

GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on 
aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting 
linear substructures of the word vector space.

GloVe is essentially a log-bilinear model with a weighted least-squares objective. The main intuition underlying the model is the simple observation that ratios of word-word co-occurrence probabilities have the potential for encoding some form of meaning. For example, consider the co-occurrence probabilities for target words ice and steam with various probe words from the vocabulary. 

TODO: Add image for glove example

As one might expect, ice co-occurs more frequently with solid than it does with gas, whereas steam co-occurs more frequently with gas than it does with solid. Both words co-occur with their shared property water frequently, and both co-occur with the unrelated word fashion infrequently. Only in the ratio of probabilities does noise from non-discriminative words like water and fashion cancel out, so that large values (much greater than 1) correlate well with properties specific to ice, and small values (much less than 1) correlate well with properties specific of steam. In this way, the ratio of probabilities encodes some crude form of meaning associated with the abstract concept of thermodynamic phase.

The training objective of GloVe is to learn word vectors such that their dot product equals the logarithm of the words' 
probability of co-occurrence. Owing to the fact that the logarithm of a ratio equals the difference of logarithms,
 this objective associates (the logarithm of) ratios of co-occurrence probabilities with vector differences in the word vector space. Because these ratios can encode some form of meaning, this information gets encoded as vector differences as well. For this reason, the resulting word vectors perform very well on word analogy tasks


For both Python Toekn Corpus and English Corpus for the custom dataset + conala dataset I have training Glove Embedding 
of 300 dimension for 100 epochs



## IV. Neurual Transformer Model

Neural transformer is based on famous paper “Attention is All You Need” https://arxiv.org/pdf/1706.03762.pdf.

Neural Transformer is based on Multi-head self attention. More details can be found in the medium article:

https://medium.com/@monimoyd/step-by-step-machine-translation-using-transformer-and-multi-head-attention-96435675be75

The Neural Transformer has Encoder and Decoder. Encoder is used for encoding input English sentences using the standard
Transformer Encoder architecture as below:

TODO: Add Enocder Architecture

For Decoder, the standard architecture is modified to input i. Python Token ii. Python Token Type and iii. Positional Embedding  
TODO: Add Decoder Architecture




## V. Loss Function

Cross-entropy loss measures the performance of a classification model whose output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label.

The cross entropy formula takes in two distributions, p(x), the true distribution, and q(x), the estimated distribution, defined over the discrete variable x and is given by

I have used Composite Cross Entropy Loss function, which has two components:

Loss1 = Cross Entroopy Loss between Predicted Python Token and Actual Python Token 
Loss1 = Cross Entroopy Loss between Predicted Python Token Type and Actual Python Token Type
 
 
 Total Loss I have used the formula:
 
 Total Loss = 1.5 * Loss1 + Loss2
 
 
## VI. Metric Used

As per Wikipedia:

BLEU (bilingual evaluation understudy) is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another.
Quality is considered to be the correspondence between a machine's output and that of a human: "the closer a machine translation is to a professional human translation, the better it is" – this is the central idea behind BLEU

I have used BLEU score for measuring the performance of generated Python Code from English Description

I have got a score of 41.61 on Test Dataset and 39.16 on Validation Dataset


## VII. Hyper parameters

The following Hyperparameters are used:

Batch Size : 128

Number of Epochs : 100

Initial Learning Rate: 0.0005


Maximum Decoder Output Length : 250

Maximum Encoder Input Length : 100

Hidden Dimension:   300

Encoder Embedding Dimension : 300

Decoder Embedding Dimension : 300

Number of Layers in Encoder: 4

Number of Layers in Decoder: 4

Number of Heads in Encoder: 6

Number of Heads in Decoder: 6

Encoder Fully Connected Layer Dimension: 1024

Decoder Fully Connected Layer Dimension: 1024

Encoder Dropout: 0.2

Decoder Dropout: 0.2



### Optimizer and Scheduler Used:

I have used Adam Optimizer and ReduceLROnPlateau scheudler with factor: 0.8 and patience: 10


## VIII. 25 Generated Python Codes

The model has been deployed in AWS EC2 and exposed through a Flask Web Application:

There are 4 fields in the screenshot

Below are screeshots of 25 generated python codes. 


i. Enter English Description to Generate Python Code - Textarea where user inputs english Description
ii. Last Query Results-> English Description - English Description of last query
iii. Last Query Results-> Generated Python Code - Generated python code of last query
iii. Last Query Results-> Program Output - Program Output  of last query.

Note: Not all prgrams generate outputs as some programs require user input, some programs need driver code to print output


### 1. Write a python program to Implement Insertion sort and print the sorted list for the below list

![program1](/python_generated_code_screenshots/program1_screenshot.png)

### 2. Write  a function that solves towers of hanoi problem

![program2](/python_generated_code_screenshots/program2_screenshot.png)

Note: - No driver code to print output hence no program output

### 3. Sum of two numbers

![program3](/python_generated_code_screenshots/program3_screenshot.png)


### 4. Write a program to find the factorial of a number

![program4](/python_generated_code_screenshots/program4_screenshot.png)


### 5. write a python function to identify the total counts of chars, digits,and symbols for given input string

![program5](/python_generated_code_screenshots/program5_screenshot.png)

### 6. Write a program to reverse a string

![program6](/python_generated_code_screenshots/program6_screenshot.png)

### 7. write a python program to print current datetime

![program7](/python_generated_code_screenshots/program7_screenshot.png)


### 8. Write a functin that returns the LCM of two input numbers

![program8](/python_generated_code_screenshots/program8_screenshot.png)


Note: - No driver code to print output hence no program output

### 9. Write a python function that prints the factors of a given number

![program9](/python_generated_code_screenshots/program9_screenshot.png)

### 10. Write a function to calculate compound interest, given p, r, t

![program10](/python_generated_code_screenshots/program10_screenshot.png)

Note: - No driver code to print output hence no program output

### 11.  write a Python program to convert Python objects into JSON strings

![program11](/python_generated_code_screenshots/program11_screenshot.png)


### 12. Write a Python function to create the HTML string with tags around the word(s)

![program12](/python_generated_code_screenshots/program12_screenshot.png)

- No driver code to print output hence no program output

### 13. Write a function to find the perimeter of a square

![program13](/python_generated_code_screenshots/program13_screenshot.png)

Note: - No driver code to print output hence no program output

### 14. Map two lists into a dictionary in Python
![program14](/python_generated_code_screenshots/program14_screenshot.png)

Note: - No driver code to print output hence no program output



### 15. write a program to convert temperature from Celsius to Fahrenheit

![program15](/python_generated_code_screenshots/program15_screenshot.png)

### 16. write a program to find and print the remainder of two number

![program16](/python_generated_code_screenshots/program16_screenshot.png)

### 17. Write a program to convert kilometers per hour to mile per hour

![program17](/python_generated_code_screenshots/program17_screenshot.png)

### 18. write a python funaction to create a new string by appending second string in the middle of first string

![program18](/python_generated_code_screenshots/program18_screenshot.png)

### 19. write a python function get the maximum number in passed list

![program19](/python_generated_code_screenshots/program19_screenshot.png)

### 20. write a program to calculate exponents of an input

![program20](/python_generated_code_screenshots/program20_screenshot.png)

### 21. write a python program that takes input a list and squares every term using list comprehension

![program21](/python_generated_code_screenshots/program21_screenshot.png)

### 22. Python Program to Find the Sum of Natural Numbers

![program22](/python_generated_code_screenshots/program22_screenshot.png)

### 23.  write a python program to check name exists in given list

![program23](/python_generated_code_screenshots/program23_screenshot.png)

### 24. zip lists in python

![program24](/python_generated_code_screenshots/program24_screenshot.png)

Note: - No driver code to print output hence no program output


### 25. write a python program to check positive number


![program25](/python_generated_code_screenshots/program25_screenshot.png)


# III. Testing

The following diagram shows the main components of training

![Project Report](/doc_images/testing_components.png)


Here inputs are
- 160x160 background images
- 160x160 foreground images superimposed on background


Processing are done by:

- Data loader loads the data. Dataloader uses the images from batch10 i.e. batch10_images.jpg
- Model is used to foreward pass through the neural network and predicts mask  


Outputs are:

- 160x160 mask of foreground on black background
- 80x80 predicted depth image


For testing, best model from training is loaded and then evaluation is done on the input images

The code for Testing is available in URL:

 https://github.com/monimoyd/PredictingMaskAndDepthUsingDeepLearning/blob/master/utils/train_test_util.py
 
 (Please look for test method)



# IV. Results

All the results are available in Jupyter Notebook 
https://github.com/monimoyd/PredictingMaskAndDepthUsingDeepLearning/blob/master/assignment15_final_api.ipynb

(Training results are in training cell output titled "Perform training of the model and periodically display statistics, mask , depth images generated, IoU value"
, and testing results are in testing cell output titled "Load the best model and perform test and Test dataset and show results")

## Results from Testing:


## i. 

### Background Image:
![test_bg_1](/results/test_bg_1.jpg)

### Foreground and Background Image:

![test_fg_bg_1](/results/test_fg_bg_1.jpg)

### Ground Truth Mask Image:

![test_fg_bg_1](/results/test_ground_truth_mask_1.jpg)

### Predicted Mask Image:

![test_fg_bg_1](/results/test_predicted_mask_1.jpg)


Predicted Mask IoU value :  0.9565858394563042

### Ground Truth Depth Image (with plasma display):

![test_fg_bg_1](/results/test_ground_truth_depth_1.jpg)

### Predicted Mask Image (with plasma display):

![test_predicted_depth_1](/results/test_predicted_depth_1.jpg)



## ii. 

### Background Image:
![test_bg_2](/results/test_bg_2.jpg)

### Foreground and Background Image:

![test_fg_bg_2](/results/test_fg_bg_2.jpg)

### Ground Truth Mask Image:

![test_fg_bg_2](/results/test_ground_truth_mask_2.jpg)

### Predicted Mask Image:

![test_fg_bg_2](/results/test_predicted_mask_2.jpg)


Predicted Mask IoU value :  0.9478972876254577

### Ground Truth Depth Image (with plasma display):

![test_fg_bg_2](/results/test_ground_truth_depth_2.jpg)

### Predicted Depth Image (with plasma display):

![test_predicted_depth_2](/results/test_predicted_depth_2.jpg)


## Result from Training

## i. 

### Background Image:
![test_bg_1](/results/train_bg_1.jpg)

### Foreground and Background Image:

![test_fg_bg_1](/results/train_fg_bg_1.jpg)


### Predicted Mask Image:

![test_fg_bg_1](/results/train_mask_1.jpg)

Mask IoU vlaue: 0.9422367689214051


### Predicted Depth Image (with plasma display):

![test_predicted_depth_1](/results/train_depth_1.jpg)




# V. Profiling:


## i. Tensorboard

TensorBoard is a profiler and visualization tool, it is  used for the following purposes:
- Tracking and visualizing metrics such as loss and accuracy
- Visualizing the model graph (ops and layers)
- Viewing histograms of weights, biases, or other tensors as they change over time

I have used  SummaryWriter from torch.utils.tensorboard  to add scalar values for metrics, IoU, Loss values for both training and testing

The tensorboard profiles are created using runs folder of Google collab. I created a tar.gz from runs folder, downloaded locally.

After unzipping started tensorboard using command:

tensorboard --logdir=runs

The output is available in http://localhost:6006

Various Tensorboard plots are as below:


### Training and Testing Loss

![loss_plot](/tensor_board_plots/loss_plot.png)


### Analysis: 

From the plot it is evident that training loss reduces in very first epoch to around 0.2 and stays flat there

The testing plot is flucutaing in a small range between 0.118 and 0.126. So loss value in test is better than training. 
This may be because I am using image augmentation during training


### Training and Testing IoU vlaues

![iou_plot](/tensor_board_plots/iou_plot.png)

From the training IoU it is evident that IoU values increases initially to around 0.92 and then it remains steady between
0.94 and 0.95

For the testing, IoU value remains steady between small range of 0.948 and 0.95


Tensor board captureed profiles files is available in URL:

https://drive.google.com/file/d/1Eye2F9UmXo_uLTueZjGGNbFiVhbxgyB8/view?usp=sharing


## ii. cProfile:

cProfile is used  for profiling Python programs. 

I have enabled cProfile by using the following lines at the beginning of Jupyter Notebook

pr = cProfile.Profile()
pr.enable()


At the end of Jupyter notebook, I have disabled cProfile and dumped the stats to a file cprofile_stats.txt

I have downloaded the file cprofile_stats.txt locally and used the cprofilev program to analyze

cprofilev -f cprofile_stats.txt

cProfile output available at http://127.0.0.1:4000

The following are some of the screenshots of cProfile:

![cprofile_plot_1](/cprofile_plots/cprofile_plot_1.png)

![cprofile_plot_2](/cprofile_plots/cprofile_plot_2.png)

![cprofile_plot_2](/cprofile_plots/cprofile_plot_2.png)

The full raw stats generated from cprofile is available in URL:


https://drive.google.com/file/d/12SyJc8aK_wlmXU2fI-_JG2cSOaW_dgdD/view?usp=sharing


## Analysis:

The train_test_utils.py line numbers 77,  24 and 29 consume lot of time. Line no 77 is related to train method, line number 24 and 29
is related to GPU profiling hooks used, which will be removed 

Another component which is consuming time is unet_model_small.py forward function in line 115 also consumes lot of time

Python libraries like tornado/stack_context.py zmq/eventloop/zmq_stream.py also consumes lot of time

Another time consuming method is tqdm/notebook.py tqdm/std.py, I can explore any lighter version available.

torch/util/data/data_loader.py and torch/util/data/_utils/fetch.py also consumes time, which can be improved by 
increasing num_workers attribute.

cProfile stats file is available in URL:

https://drive.google.com/file/d/12SyJc8aK_wlmXU2fI-_JG2cSOaW_dgdD/view?usp=sharing



## iii. GPU Profiling


I have instrumented the training code for GPU profiling based on article:

 https://www.sicara.ai/blog/2019-28-10-deep-learning-memory-usage-and-pytorch-optimization-tricks
 
 The Plots did not work. But I have converted teh memory profile to pandas dataframe and from that I have downaloaded as CSV.
 
 A few records are as below:

| layer_idx | call_idx |  layer_type              |  exp   | hook_type |    mem_all   |  mem_cached     |
| ----------|----------|--------------------------|--------| ----------|--------------|---------------- |
|    0      |    0     |    UNet                  | exp_0  |   pre     |   35588096   |   341835776     |
|    1      |    1     |  DoubleConv              | exp_0  |   pre     |   56559616   |   341835776     |
|    2      |    2     |   Sequential             | exp_0  |   pre     |   56559616   |   341835776     |
|    3      |    3     | depthwise_separable_conv | exp_0  |   pre     |   35588096   |   341835776     |
|    4      |    4     |    Conv2d                | exp_0  |   pre     |   56559616   |   341835776     |
|    4      |    5     |    Conv2d                | exp_0  |   fwd     |   77531136   |   341835776     |
|    5      |    6     |    conv2d                | exp_0  |   pre     |   77531136   |   341835776     |
|    5      |    7     |    Conv2d                | exp_0  |   pre     |   405211136  |   1000341504    |
|    3      |    8     | depthwise_separable_conv | exp_0  |   fwd     |   405211136  |   1000341504    |
|    6      |    9     |    BatchNorm2d           | exp_0  |   pre     |   405211136  |   1000341504    |

Last few records depthwise_separable_conv, BatchNorm2d are using mem_cached value of  1000341504 which is too high, needs
further attention.

The code for GPU profiling is available in URL:

https://github.com/monimoyd/PredictingMaskAndDepthUsingDeepLearning/blob/master/utils/train_test_util.py

(please look for methods _get_gpu_mem, _generate_mem_hook, _add_memory_hooks, log_mem)

Full GPU Profiling gpu_profiler_stats.csv file is available in:

https://drive.google.com/file/d/1Es4bPPQIa2937jxwIOWNi4ztVIf9zVtM/view?usp=sharing

 

## iv. Time Measurements

I have measured time copyinng zip files, unzipping the zip file, as well as training times.
 
(All measurements units are seconds)

Total time taken for copying zip files:  128.2801535129547

Total time taken for unzipping zip files:  155.29554629325867

The training time I have split into three heads:

i. Training time
ii. Data Loading time
iii. Misc time


|   Epoch      |     Training Time     |    Data Loading Time     |     Misc Time                    |
| -------------|-----------------------|--------------------------|----------------------------------|
|     1        |   3600.63835811615    |    10.604703187942505    |    56.787678956985474            |
|     2        |   3622.2514436244965  |    10.590641975402832    |    59.93659019470215             |
|     3        |   3626.0100643634796  |    10.588401794433594    |    60.445101261138916            |
|     4        |   3661.500823497772   |    10.634032964706421    |    61.81038284301758             |
|     5        |   3653.637369155884   |    10.622773170471191    |    57.52307391166687             |


### a. Epoch vs Training time Plot

![doc_images](/doc_images/epoch_vs_training_time.png)

### b. Epoch vs Data Loading time Plot

![doc_images](/doc_images/epoch_vs_dataloading_time.png)

### b. Epoch vs Misc time Plot

![doc_images](/doc_images/epoch_vs_misc_time.png)


Note: the plots are generated offline using the Jupyter Notebook:

https://github.com/monimoyd/PredictingMaskAndDepthUsingDeepLearning/blob/master/generate_plots.ipynb

## Analysis: 

From this data training time gradually increases upto around 60 seconds then sligtly decreases. Data Loading time and
Misc Time are almost constant across epochs.

Data loading time can be further reduced by changing num_workers attribute in Dataloader. 

## vi. MACs value for Model:

multiply-and-accumulate (MAC) operations gives how  better model will perform in terms of number of operations. Installed thop library
and used profile method to calculated MACs value as below:

MACs:  892294400.0

Code for calculating MACs value is in  the Jupyter Notebook:

https://github.com/monimoyd/PredictingMaskAndDepthUsingDeepLearning/blob/master/assignment15_final_api.ipynb

(Please look for cell under the title "Calculate and display MACS value of the model")


# VI. Code Structure

|  Path                                                     |        Comment                                                          |
| ----------------------------------------------------------|-------------------------------------------------------------------------|
|  assignment15_final_api.ipynb                             |  Main Jupyter Notebook for training and testing the model               |
|  data_loaders/fg_bg_images_data_loader.py                 |  Image Data Loader                                                      |
|  data_transformations/fg_bg_images_data_transformation.py |  Transformation and Augmentation                                        | 
|  utils/iou_util.py                                        |  Utility for calculating IoU                                            |
|  utils/ssim_util.py                                       |  Utility for SSIM Loss calculation                                      |
|  utils/plot_util.py                                       |  Utility for plotting images                                            |      
|  utils/train_test_util.py                                 |  Utility for training testing and GPU profiling                         |
|  results                                                  |  Image Results are stored                                               |
|  cprofile_plots                                           |  Cprofile plots are stored                                              |
|  tensorboard_plots                                        |  Tesnsorboard plots are stored                                             |

 


# VII. Problems Faced and how I addressed

## i. Loss Functions
Initilaly I was using BCELogitsLoss for both mask and depth predictions but depth image quality was not good. When I used
SSIM for depth and BCEWitLogitsLoss for Mask, I found the depth images were better and even mask images IoU is around 0.95 which 
is quite good

## ii. Unzip the batch zip image files

Initially I was trying to unzip the images zip file in google drive itself. Each batch of images were taking close to 
2 hours. So, I changed the strategy and tried copying the image zip files locally to Colab and then unzipped, it significantly
reduces the time. However, the downside is that 


## iii. Colab GPU access disabled:
My Colab account for GPU access was suspended giving reason that my usage limit is high. So changed to Colab Pro
by paying monthly subscription

## iv. Jupyter Notebook hung and colab disconnected frequently
The Jupyter notebook got hung lot of times (after say internet got disconneted for sometime). I realized that this is because I was
displaying too many imges after few iterations. So I reduced number of times image display per epoch to only two times. Even then
I faced issues, so I used the chrome extension as mentioned in telegram group post https://github.com/satyajitghana/colab-keepalive .
As I am also saving model periodically to google drive. In case I can not view Jupyter Notebook, I keep on viewing new model weights 
files are generated to know that Colab is still working on Jupyter notebook

## vi. Batch size selection
Initially I was trying bigger batch size (256,124) but I was getting out of memory. Finally I found that batch size of 100 works
without any memory issue



# VIII. Future works

## i. 
The ground truth depth images were not very high quality. My model would have given better results if ground truths are good quality.
So, as a future I will try good models to generate ground truth depth images

## ii. 
Further analysis of GPU memory profiling can be done

### iii.

 I have to run around 7.5 hours to achieve the result. As Google TPU are faster, I would like to try with TPU
 
### iv.

Apply knowledge acquired different applications like lung cancer detetion etc.



# IX. Conclusion


In this project, I have worked on predicting Mask and Depth of given background and foregroud superimposed background images. 
I have used reduced UNet model of only 748K parameters (i.e. less than 1M parameters) and predicted mask and depth almost
 closer to the ground truth values. Mask IoU is around 0.95.

I have used various profiling tools: tensorboard, cprofile, GPU profiler as well as calculated MACS value for the model.

This project is a great learning opportunity for me.













