import pandas as pd

def clean_data(df):
    
    # Drop columns
    drop_columns = [
        "id",
        "split",
        "created_date",
        "publication_id",
        "parent_id",
        "article_id",
        "male",
        "female",
        "transgender",
        "other_gender",
        "heterosexual",
        "homosexual_gay_or_lesbian",
        "bisexual",
        "other_sexual_orientation",
        "christian",
        "jewish",
        "muslim",
        "hindu",
        "buddhist",
        "atheist",
        "other_religion",
        "black",
        "white",
        "asian",
        "latino",
        "other_race_or_ethnicity",
        "physical_disability",
        "intellectual_or_learning_disability",
        "psychiatric_or_mental_illness",
        "other_disability",
        "identity_annotator_count", # this is feature engineered, making it colinear with existing variables
        "toxicity_annotator_count"  # this is feature engineered, making it colinear with existing variables
    ]
    df = df.drop(drop_columns, axis=1)

    # Drop NA Comments
    df = df.dropna()

    # Drop duplicates
    df = df.drop_duplicates()

    # Encode the rating variable
    df = pd.get_dummies(df, columns=['rating'], drop_first=True)
    df = df.rename(columns={"rating_rejection": "rating"})

    return df
