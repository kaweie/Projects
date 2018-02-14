import numpy as np
import pandas as pd
from nltk import word_tokenize, pos_tag, ne_chunk
import nltk
import ast

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

    return raw_body_text

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


def get_Num_entities(entities):
    """
    This function takes the raw format of the entities from the previous section
    and it returns the number of entities in a single post. I could technically
    do without this function at this point since I realized it is simpler to not
    consider things that I once considered.

    Input:

    entities - list produced by the format function, but after it was stripped
    from all of the non-entity inputs

    Output:

    num_entities_this_post - The number of entities in one post.

    """

    num_entities_this_post = len(entities)
    return num_entities_this_post


def main():
    posts = extract_from_file("data.csv")

    #This is a set of all named entity types
    types_set = set(["GPE","PERSON","FACILITY","GSP","LOCATION","TIME","NUMBER",'ORGANIZATION'])

    #Here I am using the format_ function to transform the posts into 
    trees = [format_(posts[j]) for j in range(len(posts))]

    num_entities = sum([get_Num_entities([trees[k][i] for i in range(len(trees[k])) if trees[k][i][0] in types_set]) for k in range(len(posts))])
    print("The number of named entities in the dataset is: ")
    print(num_entities)

main()