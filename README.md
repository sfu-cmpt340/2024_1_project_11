# Harmful Brain Activity Classification (HBAC)

This model aims to classify seizures and other patterns of harmful brain activity in critically ill patients. We'll be using data provided by Harvard Medical School, as part of an ongoing [Kaggle competition](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/overview).
## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/EZzEiGQNT3FAphR788z3IrYBEx8ASZ6C2IbP5BXt9pfhCg?e=dp9ZUx) | [Slack channel](https://app.slack.com/client/T06AP91EYG6/C06DW516NA1?selected_team_id=T06AP91EYG6) | [Project report](https://www.overleaf.com/7196966197swyzqqdqbkxt#c96211) |
|-----------|---------------|-------------------------|

## Video/demo/GIF

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/XXy3TBCCN7k/0.jpg)](https://www.youtube.com/watch?v=XXy3TBCCN7k)

## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

4. [Guidance](#guide)


<a name="demo"></a>
## 1. Example demo
For the demonstration, we created a web interface that identifies neurological disorders and displays the output to users upon receiving data input. We used Python flask, which stores our HTML files and serves as the location for our Python code. After using ```flask run``` to start a flask server, we put the single data file as a CSV file and upload the data, the analysis result shows at the bottom ['GPD,' 'GRDA,' 'LPD,' 'LRDA,' 'Other,' 'Seizure']. 


![alt text](images/demo_img.png)


### What to find where


repository
├── calculations                 ## Testing correctness for feature values
├── data                         ## EEG sample datasets 
├── images                       ## Images for our doc and demo
├── src                          ## Source code for our package itself
│   └── feature_extraction/
│   └── preprocessing/
│   └── utils/
│   └── visualize/
│
├── static/css                   ## Styling for our HTML demo
├── templates                    ## HTML templates for our demo          
├── README.md                    ## You are here
├── requirements.yml             ## Dependencies
├── hms.ipynb                    ## Jupyter notebook
├── hms.py                       ## Python file for our Jupyter notebook


<a name="installation"></a>

## 2. Installation

To clone the repository and install required dependencies please run:
```batch
git clone git@github.com:sfu-cmpt340/2024_1_project_11.git
cd 2024_1_project_11
conda env create -n 340-project-11 -f requirements.yml
conda activate 340-project-11
```

<a name="repro"></a>
## 3. Reproduction
Our python script contains code that automatically retrieves 1000 signals from our dataset.

To reproduce our model results, run the following:
```bash
conda activate 340-project-11
python hms.py
```
Alternatively, you can run the Jupyter notebook hms.ipynb.

<a name="guide"></a>
## 4. Guidance

- Use [git](https://git-scm.com/book/en/v2)
    - Do NOT use history re-editing (rebase)
    - Commit messages should be informative:
        - No: 'this should fix it', 'bump' commit messages
        - Yes: 'Resolve invalid API call in updating X'
    - Do NOT include IDE folders (.idea), or hidden files. Update your .gitignore where needed.
    - Do NOT use the repository to upload data
- Use [VSCode](https://code.visualstudio.com/) or a similarly powerful IDE
- Use [Copilot for free](https://dev.to/twizelissa/how-to-enable-github-copilot-for-free-as-student-4kal)
- Sign up for [GitHub Education](https://education.github.com/) 
