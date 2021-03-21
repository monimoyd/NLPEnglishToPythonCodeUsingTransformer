# NLPEnglishToPythonCodeUsingTransformer


# I. Problem Statement


Generating Machine code from Human Languages is a challenging propblem. In this project I have used Neural Transformer to generate Python Source Code from a given English Description. 

I have used a custom dataset where each record starts with English Description line starting with # character followed 
by python code.

The dataset is available in link below:

https://drive.google.com/file/d/116ClZ6nu1kL-RUCx-p3Sc-SGIq1zB9a-/view?usp=sharing

Jupyter Notebook link: https://github.com/monimoyd/NLPEnglishToPythonCodeUsingTransformer/blob/main/english_description_to_python_code_conversion_final.ipynb

Youtube Link: https://youtu.be/aGqa_0eroOY 

Major Highlights:

-  Use fully API based implementation 
-  Used Neural Transformer based architecture
-  In decoder used token type embedding in addition to token embedding and positional embedding
-  Built custom Python Tokenizer to tokenize python code and type
-  Used spacy English Tokenizer to tokenize english description
-  Generated Glove Embeddings for Python Code and English separately from the scratch and used these embeddings during training
-  Used a composite cross entry loss function which combines python token loss and python token type loss
-  Deployed the model in AWS EC2 and built a website using Flask

## Running the code

For Training, you can use download the code and run
git clone https://github.com/monimoyd/NLPEnglishToPythonCodeUsingTransformer.git

# II. Datasets and Cleaning

## i. Custom Python Dataset
The original custom python dataset was very messy and cluttered and it can not be processed by programs. 
Original dataset is changed manually to come up with the following format

1. English Description starts in a new line starting with # (Hash character)
2. Python Code will be immediately afte the English Description.
3. Each example pair is separated by a single blank line
4. There should not be any blank line in python code

Even after manual changes, faced some issues, written a custom parser to find the lines where the problem happens and fixed those lines

Written a loader for loading the custom Python Dataset. The following cleanups are done

i.  Any Comments in Python Code are removed
ii. Removed all the import statements in python code in dataset as there is a pyforest package available which saves the effort of generating any import statements after the package is installed.
any import statement. While generating python code just need to include "import pyforest" statement
iii. From the english description all the punctulations are removed and all the words are made lower case 

## ii. Conala Python Dataset

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

Use json loader for loading Conala dataset

## iii. Cleaning Dataset

All teh punchuations are cleaned from The combined dataset

Used Pytext and BucketIterator for generating Train, Test, Validation datasets

# III. Tokenization

For English text I ahve used spacy English tokenizer.

For Python Code, I have developed two tokenizer one for the actual tokens and second is based on token type. The custom tokenize is built on top of python built in tokenize library 
 https://docs.python.org/3/library/tokenize.html
 
 I have noted that in addition token itself teh token type also carries lot of value as while generating python the trannsformer can check right token type is generated or not. Some of teh token types are :
 i. Identifier
 ii. Keyword (like if, else, while)
 iii. Function Name
 iv. Function Declaration
 v. Operator ( e.g. =, + )
 vi. New Line
 vii. Indent
 viii. Deindent
 
 In addition to token itself, I have used token type as a feature, created embedding. Also, in loss function, I have used composite function involving actual token as well as token type
 
 

# IV. Data Augmentation

I have used the following Data Augmentation Techniques


## 1. Synonym Replacement

First, we could replace words in the sentence with synonyms, like so:

The dog slept on the mat

could become

The dog slept on the rug

Aside from the dog's insistence that a rug is much softer than a mat, the meaning of the sentence hasn’t changed. But mat
and rug will be mapped to different indices in the vocabulary, so the model will learn that the two sentences map to the
same label, and hopefully that there’s a connection between those two words, as everything else in the sentences is the same
 

## B. Random Swap

The random swap augmentation takes a sentence and then swaps words within it n times, with each iteration working on the
previously swapped sentence. Here we sample two random numbers based on the length of the sentence, and then just keep
swapping until we hit n.
 

## C. Random Deletion

As the name suggests, random deletion deletes words from a sentence. Given a probability parameter p, it will go through the
sentence and decide whether to delete a word or not based on that random probability. Consider of it as pixel dropouts 
while treating images. 
 
 
### D. Back Translation

Back Translation involves translating a sentence from our target language into one or more other languages and then translating all of them back to the original language. 
I have  used google_trans_new Python library for this purpose. Note that google_trans_new Python library is very slow, so I used
only 5 percent of my total training set for performing Back Translation

# IV Glove Embedding

{Source: https://nlp.stanford.edu/projects/glove/]

GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on 
aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting 
linear substructures of the word vector space.

GloVe is essentially a log-bilinear model with a weighted least-squares objective. The main intuition underlying the model is the simple observation that ratios of word-word co-occurrence probabilities have the potential for encoding some form of meaning. For example, consider the co-occurrence probabilities for target words ice and steam with various probe words from the vocabulary. 

![Glove](/docs/glove_example.png)

As one might expect, ice co-occurs more frequently with solid than it does with gas, whereas steam co-occurs more frequently with gas than it does with solid. Both words co-occur with their shared property water frequently, and both co-occur with the unrelated word fashion infrequently. Only in the ratio of probabilities does noise from non-discriminative words like water and fashion cancel out, so that large values (much greater than 1) correlate well with properties specific to ice, and small values (much less than 1) correlate well with properties specific of steam. In this way, the ratio of probabilities encodes some crude form of meaning associated with the abstract concept of thermodynamic phase.

The training objective of GloVe is to learn word vectors such that their dot product equals the logarithm of the words' 
probability of co-occurrence. Owing to the fact that the logarithm of a ratio equals the difference of logarithms,
 this objective associates (the logarithm of) ratios of co-occurrence probabilities with vector differences in the word vector space. Because these ratios can encode some form of meaning, this information gets encoded as vector differences as well. For this reason, the resulting word vectors perform very well on word analogy tasks

For both Python Token Corpus and English Corpus for the custom dataset + conala dataset I have training Glove Embedding 
of 300 dimension for 100 epochs and loaded as a pretrained embedding while training


## IV. Neural Transformer Model

Neural transformer is based on famous paper “Attention is All You Need” https://arxiv.org/pdf/1706.03762.pdf.

Neural Transformer is based on Multi-head self attention. More details can be found in the medium article:

https://medium.com/@monimoyd/step-by-step-machine-translation-using-transformer-and-multi-head-attention-96435675be75

The Neural Transformer has Encoder and Decoder. Encoder is used for encoding input English sentences using the standard
Transformer Encoder architecture as below:

![Encoder](/docs/Encoder.png)

The encoder takes the english words, create embedding which is combined with positional embedding and then layers of transformers are applied to get the encoded representation

For Decoder, the standard architecture is modified to input i. Output Python Token ii. Ouptut Python Token Type and iii. Position, the embedding are created for all these three and then passed throuhg masked multihead attention layers, which is then combined with encoder output and passed  through teh multihead attention layers to get the final output

![Decoder](/docs/TransformerNewDecoder.png)



## V. Loss Function

Cross-entropy loss measures the performance of a classification model whose output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label.

The cross entropy formula takes in two distributions, p(x), the true distribution, and q(x), the estimated distribution, defined over the discrete variable x and is given by

I have used Composite Cross Entropy Loss function, which has two components:

Loss1 = Cross Entroopy Loss between Predicted Python Token and Actual Python Token 
Loss2 = Cross Entroopy Loss between Predicted Python Token Type and Actual Python Token Type
 
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


# VIII. 25 Generated Python Codes

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



# X. Results: Comparion of predicted and actual actual and attention


## i. Example from Validation dataset
English Description: 
arrange string characters such that lowercase letters should come first

### Predicted Python Code:

import pyforest
str1 = "PyNaTive"

lower = [ ]

upper = [ ]

for char in str1 :

    if char.islower() :
    
        lower.append(char)
        
    else :
    
        upper.append(char)
        
sorted_string = ''.join(lower + upper)

print(sorted_string)


### Actual Python Code:

str1 = "PyNaTive"

lower = [ ]

upper = [ ]

for char in str1 :

    if char.islower() :
    
        lower.append(char)
        
    else :
    
        upper.append(char)
        
sorted_string = ''.join(lower + upper)

print(sorted_string)

### Attention:

![val1_attention1](/docs/val1_attention1.png)
![val1_attention2](/docs/val1_attention2.png)
![val1_attention3](/docs/val1_attention3.png)
![val1_attention4](/docs/val1_attention4.png)
![val1_attention5](/docs/val1_attention5.png)
![val1_attention6](/docs/val1_attention6.png)

## i. Example from Test dataset

English Description: 
write python3 program for illustration of values method of dictionary

### Predicted Python Code:

import pyforest

test_dict = { 'gfg' : True , 'is' : False , 'best' : True }

print("The original dictionary is : " + str(test_dict))

res = True

for key , value in test_dict.items() :

    if key in res.items() :
    
        res = False
        
        break
        
print(f"Dictionary is {res}")


### Actual Python Code:

dictionary = { "raj" : 2 , "striver" : 3 , "vikram" : 4 }
print(dictionary.values())

### Attention:

![test1_attention1](/docs/test1_attention1.png)
![test1_attention2](/docs/test1_attention2.png)
![test1_attention3](/docs/test1_attention3.png)
![test1_attention4](/docs/test1_attention4.png)
![test1_attention5](/docs/test1_attention5.png)
![test1_attention6](/docs/test1_attention6.png)


# X. Metrics

Various Metrics generated on validation and test datasets are as below:

Validation Loss: 1.444

Validation PPL:   4.236

Validation BLEU score = 39.16

Test Loss: 1.619 

Test PPL:   5.046

Test BLEU score: 41.61

# XI. Plots of Loss and PPL

The plot of Loss values for Train and validation datasets over epochs is as below:

![loss_plot](/docs/train_val_loss_plot.png)


The plot of PPL values for Train and validation over epochs is as below:

![ppl_plot](/docs/train_val_ppl_plot.png)

# XII. Issues faced and how I addressed

## 1. torchtext ImportError in colab
One day, I suddently saw none of notebooks were working because of import error in torchtext. This is because of version in Colab got matched

Donwngraded to version below:

!pip install -U torch==1.7.0
!pip install -U torchvision==0.8.1
!pip install -U torchtext==0.8.0

Other solution is to use torchtext.legacy which I used later


## 2. CUBLAS_STATUS_ALLOC_FAILED error :

I got the Cuda error below

RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`

I read some blogs and it was suggested to run on CPU after running on CPU:

After lot of debugging I found that it was becuase of embedding dimesion issue. Basically in the decoder I have declared 
maxlength as 100 but some cases, the training length exceeding 100 as a result position embedding fails
  
Links used:
https://stackoverflow.com/questions/56010551/pytorch-embedding-index-out-of-range

## 3. device-side assert triggered :

RuntimeError: transform: failed to synchronize: cudaErrorAssert: device-side assert triggered

For each dataset I was checking if teh target sequence length is less than MAX_OUTPUT_SEQ_LENGTH, but torchtext adds 4 more
tokens, so length must be restricted to MAX_OUTPUT_SEQ_LENGTH-4

## 4. '<' not supported :

TypeError: '<' not supported between instances of 'Example' and 'Example'

The issue happens as no operator is defined in sort, so I added a sort field while populating iterator




# IX. Conclusion


In this project, I have worked on predicting Mask and Depth of given background and foregroud superimposed background images. 
I have used reduced UNet model of only 748K parameters (i.e. less than 1M parameters) and predicted mask and depth almost
 closer to the ground truth values. Mask IoU is around 0.95.

I have used various profiling tools: tensorboard, cprofile, GPU profiler as well as calculated MACS value for the model.

This project is a great learning opportunity for me.













