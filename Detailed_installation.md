This guide is designed for users who may not be familiar with coding or command-line tools. We will set up a virtual environment to keep all the software for EEG-Pype organized and install the necessary software step-by-step.
Note: You only need to do steps 1 through 4 once.

# 1. Install the necessary tools
## A. Install Miniconda
Miniconda is a tool that manages the Python coding language for us.
- Windows: Download the [Miniconda Windows Installer (64-bit)](https://docs.conda.io/en/latest/miniconda.html#windows-installers). Open the file and click "Next" through the installation. Important: If asked, leave the default settings as they are.
- macOS:
  - If you have a newer Mac (M1, M2, M3 chips): Download the macOS Apple M1 64-bit installer.
  - If you have an older Mac (Intel chip): Download the macOS Intel x86 64-bit installer.
  - Link to [macOS Installers](https://docs.conda.io/en/latest/miniconda.html#macos-installers).

## B. Install Git
Git allows you to download the software from this website.
•	Windows: Download and install Git for Windows. During installation, you can keep clicking "Next" to accept all default settings.
•	macOS: Git is usually installed automatically on Mac computers. To check, you can skip this step for now; if you need it later, your Mac will prompt you to install it.

# 2. Download the EEG-Pype Code
## A.	Open your command tool:
o	Windows: Open Anaconda Prompt (Miniconda3) (Do not use the standard Command Prompt/cmd).
o	macOS: Open Terminal.

## B. Copy and paste the following command into that window and press Enter:
Bash
git clone https://github.com/yorbenlodema/EEG-Pype.git

This downloads the EEG-Pype folder to your user folder.

# 3. Create the Environment

Now we will create the virtual environment or "sandbox" where the software lives.
## A.	Enter the folder: Copy and paste this command and press Enter:
Bash
cd EEG-Pype

## B.	Create the environment: Copy and paste this command and press Enter.
o	Note: This step requires internet and may take several minutes. You will see lines of text scrolling; this is normal.
Bash
conda env create -f Environment.yml

## C.	Activate the environment: We need to tell the computer to enter the virtual environment we just made. Run this command:
Bash
conda activate EEG-Pype

Check: Look at the left side of your command line. It should now say (EEG-Pype) instead of (base).

# 4. Install the Interface (PySimpleGUI)
Due to licensing changes, this specific part of the interface needs to be installed separately.
## A.	Make sure your command line still says (EEG-Pype) on the left.
## B.	Copy and paste this exact line and press Enter:
Bash
python -m pip install --force-reinstall --extra-index-url https://PySimpleGUI.net/install PySimpleGUI

# 5. Running the Software
Now that everything is installed, here is how you open the software.

## A.	Open Anaconda Prompt (Miniconda3) (Windows) or Terminal (macOS).
## B.	Activate your environment by typing:
Bash
conda activate EEG-Pype

## C.	Navigate to the source folder:
Bash
cd EEG-Pype/src

## D.	Run the preprocessing program:
Bash
python eeg_processing_script.py

Or run the quantitative analysis module:
Bash
python eeg_quantitative_analysis.py


You can also choose to open and edit the scripts included in EEG-Pype in a code editor like Spyder or Visual Studio Code. While running the scripts from the command line gives the most stable performance, 
the eeg_processing_script.py script can also be run from your code editor. For the eeg_quantitative_analysis.py script, we have found that currently, only running from the command line is possible.

# Summary for Daily Use
Next time you want to use the software, you only need to do this:
1.	Open Anaconda Prompt (Windows) or Terminal (Mac).
2.	Type: conda activate EEG-Pype
3.	Type: cd EEG-Pype/src
4.	Type: python eeg_processing_script.py

