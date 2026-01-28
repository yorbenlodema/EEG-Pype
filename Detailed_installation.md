This guide is designed for users who may not be familiar with coding or command-line tools. We will set up a virtual environment to keep all the software for EEG-Pype organized and install the necessary software step-by-step.
Note: You only need to do steps 1 through 4 once.

# 1. Install the necessary tools
## A. 
Install Miniconda:
Miniconda is a tool that manages the Python coding language for us. [Miniconda Download link](https://www.anaconda.com/download/success).
- Windows: Under "Windows" download the "Windows 64-Bit Graphical Installer". Open the file and click "Next" through the installation. Important: If asked, leave the default settings as they are.
- macOS:
  - If you have a newer Mac (M1-M5 chips): Under "Mac" and "Miniconda" download the 64-Bit (Apple silicon) Graphical Installer.
  - If you have an older Mac (Intel chip): Under "Mac" and "Miniconda" download the 64-Bit (Intel chip) Graphical Installer.


## B. 
Install Git:
Git allows you to download the software from this website.
- Windows: Download and install [Git for Windows](https://git-scm.com/download/win) ("Standalone Installer" then "x64" for most users unless you have an ARM-based CPU). During installation, you can keep clicking "Next" to accept all default settings.
- macOS: Git is usually installed automatically on Mac computers. To check, you can skip this step for now; if you need it later, your Mac will prompt you to install it.

# 2. Download the EEG-Pype Code
## A.	
Open your command tool:
- Windows: Open Anaconda Prompt (Miniconda3) (Do not use the standard Command Prompt/cmd).
- macOS: Open Terminal.

## B. 
Copy and paste the following command into that window and press Enter:
```Bash
git clone https://github.com/yorbenlodema/EEG-Pype.git
```

This downloads the EEG-Pype folder to your user folder.

<img width="732" height="553" alt="SCR-20260128-oonz" src="https://github.com/user-attachments/assets/6294b9fb-dc61-40ef-98f0-b166c2ec10f0" />
<img width="732" height="553" alt="SCR-20260128-oozi" src="https://github.com/user-attachments/assets/1d22c41d-c8ed-4bd6-b626-55406647508d" />


# 3. Create the Environment

Now we will create the virtual environment or "sandbox" where the software is installed. This environment contains all packages (dependencies) that EEG-Pype relies on.
## A.	
To enter the correct folder, copy and paste this command and press Enter:
```Bash
cd EEG-Pype
```
<img width="732" height="553" alt="SCR-20260128-opcz" src="https://github.com/user-attachments/assets/34048fd6-033d-48ce-9d75-92b45abb03ac" />

## B.	
To create the virtual environment, copy and paste this command and press Enter:

```Bash
conda env create -f Environment.yml
```
Note: This step requires internet and may take a little while. You will see lines of text scrolling; this is normal.

<img width="732" height="553" alt="SCR-20260128-opfm" src="https://github.com/user-attachments/assets/6989c322-db47-421f-8504-75c6eef67144" />

## C.	
To activate the environment we just made, run this command:
```Bash
conda activate EEG-Pype
```

Check: Look at the left side of your command line. It should now say (EEG-Pype) instead of (base). 

<img width="816" height="623" alt="SCR-20260128-oppy" src="https://github.com/user-attachments/assets/4eabec0e-9c5b-4300-a823-96f8090e3dfa" />

# 4. Install the Graphical User Interface (PySimpleGUI). 
Due to licensing changes, this specific part of the interface needs to be installed separately.
## A.	
Make sure your command line still says (EEG-Pype) on the left.
## B.	
Copy and paste this exact line and press Enter:
```Bash
python -m pip install --force-reinstall --extra-index-url https://PySimpleGUI.net/install PySimpleGUI
```

# 5. Running the Software
Now that everything is installed, here is how you open the software. Make sure in Anaconda Prompt (Windows) or Terminal (Mac) you see (EEG-Pype) as the first thing on the last line, instead of (base). Seeing (EEG-Pype) here means that we have moved to or activated our virtual environment, containing the packages/dependencies EEG-Pype needs.

## A.	
Open Anaconda Prompt (Miniconda3) (Windows) or Terminal (macOS).
## B.	
Activate your environment by typing:
```Bash
conda activate EEG-Pype
```

## C.	
Navigate to the source folder:
```Bash
cd EEG-Pype/src
```

Or if you were already in the EEG-Pype folder:
```Bash
cd src
```

## D.	
Run the preprocessing program:
```Bash
python eeg_processing_script.py
```

<img width="858" height="665" alt="SCR-20260128-oqvq" src="https://github.com/user-attachments/assets/0858e148-99a6-4163-885d-a02344544c74" />
<img width="1112" height="915" alt="SCR-20260128-oqyq" src="https://github.com/user-attachments/assets/e9822c4c-600f-4301-8d07-7f29a271ed42" />

Or run the quantitative analysis module:
```Bash
python eeg_quantitative_analysis.py
```

<img width="858" height="665" alt="SCR-20260128-orab" src="https://github.com/user-attachments/assets/62c169e6-d208-4f61-bb84-5bc5bf5553dc" />
<img width="861" height="857" alt="SCR-20260128-orbk" src="https://github.com/user-attachments/assets/e381854b-0b0f-4d24-a0ca-77c174f70607" />

You can also choose to open and edit the scripts included in EEG-Pype in a code editor like Spyder or Visual Studio Code. While running the scripts from the command line gives the most stable performance, the eeg_processing_script.py script can also be run from your code editor. For the eeg_quantitative_analysis.py script, we have found that currently, only running from the command line is possible.

# Summary for Daily Use
Next time you want to use the software, you only need to do this:
1.	Open Anaconda Prompt (Windows) or Terminal (Mac).
2.	Type: conda activate EEG-Pype
3.	Type: cd EEG-Pype/src
4.	Type: python eeg_processing_script.py

