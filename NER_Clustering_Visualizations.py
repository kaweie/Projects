import numpy as np
import pandas as pd
from nltk import word_tokenize, pos_tag, ne_chunk
import nltk
import ast
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_from_file(filename):
    """
    This function extracts a csv file and returns a
    pandas dataframe that contains all of the twitter
    posts contained on the csv file.

    Input:

    filename - A string with the filename of the csv file.

    Output:

    raw_body_text - A pandas dataframe with just the text files
    """

    data_table = pd.DataFrame.from_csv("data.csv")

    #This is line is just extracting the raw twitter posts
    raw_body_text = data_table['raw_body_text']
    
    #sentiment_category = data_table['sentiment']
    author_follower_count = data_table['author_followers_count']
    is_reshare = data_table['is_reshare']
    loc = data_table['location']
    return raw_body_text,author_follower_count, is_reshare, loc

def format_(post):
    """
    This function takes the pandas data frame from
    previous function and turns it into list that is 
    easier to extract specific information. 

    This function actually performs the NER function 
    of the project.

    Input:

    post - This is a string that is a twitter post.

    Output:

    tree - This is an nltk tree from the nltk library.
    This tree is formatted to show how each word in 
    post is classified. Later on we will take out all
    of the unnamed entities, but we haven't gotten there
    yet.
    """

    entities = ne_chunk(pos_tag(word_tokenize(post)))
    tree = repr(entities)
    tree = (("[" + tree[5:-1] + "]")).replace("Tree", "").replace(")", "]").replace("(", "[")
    tree = ast.literal_eval(tree)
    tree = tree[1]
    return tree

def clustering_algorithm():
    """
    This function is meant to exercise the above functions to generate a data set and to the apply
    a machine learning algorithm from scikit learn. I used very similar code here to the find and
    count part of the project. The problem with this however is that I decided to get rid of the
    double list comprehension in this code. It takes significantly longer for this function to compile
    because it was necessary for it to compile.
    
    Output: 
    
    X - A Pandas DataFrame that contains data that will be visualized in the next function. 
    
    This function is supposed to print out a statement about the accuracy of the algorithm based on the data.
    
    In this function, I use a K-Nearest Neighbors algorithm to learn how reshares and location may affect
    the number of followers. I felt that the instructions to "use a clustering algorithm" was a little vague
    so I decided to classify information based on these variables.
    """
    types_set = set(["GPE","PERSON","FACILITY","GSP","LOCATION","TIME","NUMBER",'ORGANIZATION'])
    posts, num_followers, is_reshare, loc = extract_from_file("data.csv")

    trees = [format_(posts[j]) for j in range(len(posts))]


    num_entities = []
    list_entities = []

    for k in range(len(posts)):
        for i in range(len(trees[k])):
            num = 0
            if trees[k][i][0] in types_set:
                num += 1
                list_entities.append(trees[k][i])


        num_entities.append(num)

    is_reshare = [int(is_reshare[i]) for i in range(len(is_reshare))]

    #This is simply called again for convenience
    data_table = pd.DataFrame.from_csv("data.csv")

    #Getting the location
    loca = list(data_table['location'])
    longs = []
    lats = []

    for entry in loca:
        word = entry.split(" ")
        longs.append(float(word[1]))
        lats.append(float(word[3]))

    #Convertiong the lists into a numpy array
    array = np.array([num_followers,is_reshare,longs,lats]).T

    """
    My dependent variable is the number of named entities
    My independent variables are the number of followers, whether or not it was reshared, longitude, and latitude.
    """
    X = pd.DataFrame(data=array,columns=["Followers","Reshared","Longitude","Latitude"])
    Y = pd.DataFrame(data = np.array(num_entities).T)

    #Splitting the data into test and training data at random
    X_train, X_test, y_train, y_test = train_test_split(X,Y)
    neigh = KNeighborsClassifier()
    neigh.fit(X_train, y_train)
    score = neigh.score(X_test,y_test)
    print("The accuracy score comparing the training and test data is:", round(score,4)*100.,"% accurate")
    return X

def visualize_data(X):
    """
    This function simply visualizes some of the data from the function 'clustering_algorithm'. This
    function completes the objective "Visualize the Data"
    
    Input:
    X - A Pandas DataFrame that contains the information in which we will visualize. This was generated
    in the function 'clustering algorithm.
    
    """
    
    
    
    """
    This next part plots the Longitude and Latitude of twitter posts. This isn't the best visualization because
    I'm sure there is a way to show the world map as a background. Since visualization is a sub-objective
    and I focused more on the NER part of the project. By looking at the visualization you can tell that 
    most of the users are in North American and Europe.
    """

    plt.plot(X['Longitude'],X['Latitude'],'r,')
    plt.title("Location of Twitter Posts: Latitude and Longitude")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    """
    This plots the Longitude and Latitude of the twitter posts along with the number of followers
    of the posts. This visualization one makes sense if you can look at it from multiple angles.
    It also appears that the regions with the most users with the most followers is probably North
    America.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X['Longitude'],X['Latitude'],zs = X['Followers'], c = 'r')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Number of Followers')
    plt.title("Location in the World and Number of Followers")
    plt.show()

def main():
    X = clustering_algorithm()
    visualize_data(X)

main()