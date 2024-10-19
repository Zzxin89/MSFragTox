# **MSFragTox**

![image](https://github.com/Hanxiaoxiao123/MSFragTox/assets/128465539/fc172335-ddae-480f-b445-a4c47bf64cd9)

# **Training of MSFragTox model**

We collected the in vitro toxicity data from Tox21 and Tox21 Challenge, and the tandem mass spectrometry (MS/MS) data from GNPS, 7 endpoints related to endocrine disruption. Then we matched the toxicity and MS/MS data to get their shared compounds, corresponding MS/MS data, and toxicity labels. The MS/MS data was first calculated by SIRIUS to get the molecular fingerprints. And we integrated the fingerprints from positive/negative mode to equal-length vectors. For each endpoint, we got a data matrix consisting of fragment probability vectors. To train the XGBoost model for each endpoint, data division according to compounds, oversampling on compounds, SMOTE on fragment probability vectors, parameter tuning with Optuna were conducted to get the optimal model for each endpoint. The code for model training and test, and models for 7 endpoints can be found in the folder named “MSFragTox_model_training”.


# **How can our models predict the toxicity of unknown MS/MS?**

## **Requirements:**
SIRIUS installed (Installation - SIRIUS Documentation (boecker-lab.github.io)); 

Python, and related packages (pandas, xgboost) should be installed (It is recommended that these are installed in a separate environment). 


## **Step 1: use SIRIUS to calculate fingerprints.**
Import the MS/MS file (mfg format for example) to into SIRIUS, calculate the formula and fingerprints, and output the result project to a specific directory. You can use the **Graphical User Interface** or **Command Line Interface**.

**Take the command line interface for example:**

(1) Say the directory of MS/MS mgf file is 
_"D:\GitHub\MSFragTox\MSFragTox_prediction\test examples\test_6ppdq\test_6ppdq.mgf"_ 

(2) set the output directory is 
_"D:\GitHub\MSFragTox\MSFragTox_prediction\test examples\test_6ppdq\6ppdq SIRIUS result"_

(3) change directory to the folder containing SIRIUS.exe, and use the following command to calculate fingerprints.

    sirius -i "D:\GitHub\MSFragTox\MSFragTox_prediction\test examples\test_6ppdq\test_6ppdq.mgf" -o "D:\GitHub\MSFragTox\MSFragTox_prediction\test examples\test_6ppdq\6ppdq SIRIUS result" formula fingerprint

## **Step 2: predict toxicity of 7 endpoints**   

Change directory to the fold containing "MSFragTox_predict.py", input the SIRIUS output project directory and result output path (optional) and wait for the prediction results! If there is no output directory set, the prediction result will show in the command line interface. And you can use _**python MSFragTox_predict.py –help**_ to see the user guide.



### Example: 

**Command:**

    python MSFragTox_predict.py -i "D:\GitHub\MSFragTox\MSFragTox_prediction\test examples\test_6ppdq\6ppdq SIRIUS result" -o "D:\GitHub\MSFragTox\MSFragTox_prediction\test examples\test_6ppd"

**Output result in the file:**


    Prediction result1
    Toxicity endpoints: Aromatase,AhR,AR,ER,GR,TSHR,TR
    Predicted toxicity possibility:
    0.2919416,0.60688066,0.04539475,0.2467532,0.07547763,0.08442728,0.20271379
    Prediction result: inactive,active,inactive,inactive,inactive,inactive,inactive


# **License**  

This project is licensed under the MIT License - see the LICENSE.md file for details.


