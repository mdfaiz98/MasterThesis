The csv files of data is in the sciebo link below-   
[Modified CSV Files](https://tu-dortmund.sciebo.de/s/3gHYDKccvbz32cQ)


They contain the data of Participants 5 - 19. The other 5 participant data was not considered for various reasons.

The raw data was first converted to CSV files using [rosbag_to_csv converter](https://github.com/AtsushiSakai/rosbag_to_csv).

Other information required for the data analysis were added as well.
The modified CSV files contains the [Experiment Id](/Experimental%20Design/Task%20Naming.png),the particpant id and the specific weighted TLX scores aswell. 

[The Final Stress Detection Algorithm](./Final%20Stress%20Detection%20Algorithm.py) script extracts the features from all the 15 participants as well uses the NASA-TLX Value to label the stress data aswell. It outputs a combine data file which can be used for the data analysis and model training using [Final Stress_Model](Final%20Stress_model%20.ipynb). 
Use python 3.9 and most of the dependencies are listed in the [requirements.txt](/Stress%20Detection/requirements.txt) file.

