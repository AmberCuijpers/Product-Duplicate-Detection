import re
import numpy as np
import pandas as pd


def create_dataframe(data):
    dataframe = []
    for key, values in data.items():
        for product in values:
            # Resolution is added but not used in this paper. However, for further research can be interesting to use
            resolution = product.get('featuresMap').get('Recommended Resolution') or product.get('featuresMap').get('Vertical Resolution')
            dataframe.append({
                'modelID':key,
                'title': product.get('title'),
                'shop': product.get('shop'),
                'brand': product.get('featuresMap').get('Brand'),
                'resolution': resolution
            })

    return dataframe


def completing_dataframe(dataframe):
    new_dataframe = pd.DataFrame(dataframe, columns=['modelID', 'title', 'shop', 'brand', 'resolution'])
    unique_brands = set()
    unique_brands.update(["TCL", "Insignia", "Avue", "Optoma", "Venturer", "Dynex", "Mitsubishi", "CurtisYoung", "Azend Group", "Hiteker", "Contex", "ProScan", "GPX", "Viore", "Elite"])
    unique_resolutions = set()

    for brand in new_dataframe['brand'].dropna().unique():
        # Make a list of all unique brands, so no brand is mentioned twice
        unique_brands.add(brand)

    # Iterate through the brands that still have value None, and search for the brand in the title to add this
    for i, row in new_dataframe[new_dataframe['brand'].isnull()].iterrows():
        if row['brand'] is None:
            # If there is no brand in the brand column, then look at the title if the brand is mentioned there
            title = row['title']
            matching_brands = [brand for brand in unique_brands if brand in title]
            if matching_brands:
                new_dataframe.at[i, 'brand'] = matching_brands[0]

    for resolution in new_dataframe['resolution'].dropna().unique():
        unique_resolutions.add(resolution)

    for i, row in new_dataframe[new_dataframe['resolution'].isnull()].iterrows():
        if row['resolution'] is None:
            title = row['title']
            matching_resolutions = [resolution for resolution in unique_resolutions if resolution in title]
            if matching_resolutions:
                new_dataframe.at[i, 'resolution'] = matching_resolutions[0]
            else:
                # If the resolution is not given for this product, we give it all the possible resolution values
                new_dataframe.at[i, 'resolution'] = unique_resolutions

    missing_brands(new_dataframe)
    missing_resolutions(new_dataframe)
    return new_dataframe


# Print brands of the products which still have value None as brand, so we can add these manually
def missing_brands(dataframe):
    for i in range(len(dataframe)):
        if dataframe['brand'][i] is None:
            print("Missing brand: ", dataframe['title'][i])


# Print the resolutions of the products which still have None as resolution, so we can adjust these manually
def missing_resolutions(dataframe):
    for i in range(len(dataframe)):
        if dataframe['resolution'][i] is None:
            print("Missing resolution: ", dataframe['title'][i])


def cleaning_data(dataframe):
    # All titles, brands and shops to lower letters
    dataframe['title'] = dataframe.apply(lambda row: clean_title(row['title'], dataframe), axis=1)
    dataframe['title'] = dataframe['title'].str.strip()

    dataframe['brand'] = dataframe['brand'].str.lower()
    dataframe['shop'] = dataframe['shop'].str.lower()
    return dataframe


def clean_title(title, dataframe):
    inches = ["'", '"', "inches", " inch", "-inch", '”', "\""]
    hertz = ["hertz", " hz", "-hz", " - hz"]
    websites = ["amazon.com", "bestbuy.com", "best buy", "newegg.com", "thenerds.net"]

    title = title.lower()   # Convert to lowercases

    for i in range(len(dataframe)):
        title = re.sub("[^a-zA-Z0-9\s\.]", "", title)

        for inch in inches:
            title = title.replace(inch, "inch")

        for hz in hertz:
            title = title.replace(hz, "hz")

        for website in websites:
            title = title.replace(website, "")
    return title


def duplicates_counter(dataframe):
    num_entries = len(dataframe)
    # Initialize a matrix to store duplicates, where the value is 1 if row and column are the same
    duplicate_matrix = np.zeros((num_entries, num_entries), dtype=int)
    duplicates = 0
    for i in range(num_entries):
        model_id_i = dataframe['modelID'][i]
        for j in range(i + 1, num_entries):
            model_id_j = dataframe['modelID'][j]
            # Duplicates if their modelIDs are the same
            if model_id_i == model_id_j:
                duplicate_matrix[i, j] = 1
                duplicate_matrix[j, i] = 1
                duplicates += 1
    return duplicate_matrix, duplicates

