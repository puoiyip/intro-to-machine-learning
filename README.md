# CardioTree (Heart Disease Predictor)
## How to Run

## How the Project was Made
This project was made primarily by following the instructions from course 3 of Girls Who Code's Data Science & AI track: Intro to Machine Learning. It was first made using GWC's TextJam IDE (hence why I pasted a huge chunk of code on my first commit). This model uses Python and several libraries including Pandas, Matplotlib, and Scikit Learn to analyze, graph, and train the machine learning model. 

This is my first major Python project, so I struggled a lot with deploying it, especially since there are multiple ways to deploy Python projects. For this project, I chose to use Streamlit to most effectively visualize my data. I also used the joblib library to export certain parts of my data for use in the app. 

I used many tutorials to understand and figure out how to deploy my project. Based on the tutorials I looked at, I wanted to actually *use* my model to make predictions on the development of heart disease based on user inputs. I was able to figure out how to make a new dataframe using user inputs, but then I got stuck on encoding the dataframe in the same manner that the model's data was encoded. After encountering this struggle, I ended up just displaying the data and the accuracy of the model, just like the original GWC project intended to do (hence why there's a huge chunk of code commented out in app.py. I plan on fixing this issue in the near future).
## How it Works
This project is a pretty basic machine learning model. I used a script from GWC to encode the data and used the "train_test_split" function from the Scikit Learn library to split the data into training and testing datasets. I then used a decision tree classifier to classify the data (heart disease vs no heart disease) based on the factors listed in the data. Afterwards, I simply calculated the accuracy of both datasets. Finally, I exported the parts of the code I needed using joblib.dump().

All of this information is displayed using Streamlit. I loaded the data I needed using joblib and displayed the dataframes, bar chart, and model accuracies using Streamlit functions.
## Resources
**About the dataset:**

https://assets.ctfassets.net/idmfh0ehr6d7/4IL3gEMHTzx9bMDknyZg1z/e226c0a99a61c606fe2daca5f5825552/kaggle-datasets-kamilpytlak-personal-key-indicators-of-heart-disease.pdf

https://assets.ctfassets.net/idmfh0ehr6d7/1lX30HE8N4hthfGDzU95lS/d3563cad2f31bdd20a396439660b42e1/heartDisease_2020_sampling_-_heartDisease_2020_sampling.pdf

**API functions**: 

https://docs.streamlit.io/develop/quick-reference/cheat-sheet

**Help with exporting data from my main.py file:**

https://www.usdsi.org/data-science-insights/want-to-deploy-your-ml-projects-with-pandas-and-scikit-learn


**Examples that I worked off of to understand how deployment works:**

https://medium.com/@adeniyi221/how-to-deploy-your-machine-learning-model-using-streamlit-455b67a69e31

https://medium.com/@benjaminkipkem/building-a-machine-learning-model-with-streamlit-8e28ea8516a0