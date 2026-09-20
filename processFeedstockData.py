import pandas as pd
import numpy as np
import os 

##################################################
### BFL data, tied to BILLION-TON data
### source: https://bioenergylibrary.inl.gov/library/?url=Data/Dataset.aspx
### read
df_raw = pd.read_excel("Inputs/DOE/BFLexport_CompAnal_FuelProperties_202108038.xlsx", 
                                   sheet_name="Sample Export", 
                                   header=[0, 1, 2, 3],#skiprows=3
                                   )

## get info
df_info = df_raw["Sample Information"]
# remove headers
df_info.columns = df_info.columns.droplevel(0)
df_info.columns = df_info.columns.droplevel(0)
# get info we want
info_columns_keep = ["State", "County", 
                     "BT16-Resource Category",
                     "BT16-Resource Sub-Category",
                     "CropType",
                     "Sub-Classification",
                     "Sample Classification",
                     "Unique Sample Type"]
# get only these columns
df_info_red = df_info[info_columns_keep]
# check: df_info_red[df_info_red["CropType"] != df_info_red["Unique Sample Type"]]


## get fuel analyses, then composition
df_fuelAnalyses = df_raw["Analyses"]["Fuel Properties Characterization"]

# get HHV 
df_HHV = df_fuelAnalyses["Calorific Analysis (ASTM D 5865-10a)"]
df_HHV = df_HHV.copy()
# convert HHV from BTU/lb to MJ / kg; source: https://www.fao.org/4/T0269e/t0269e0a.htm
BtupLb_to_MJpKg = 1 / 430
df_HHV["HHV, MJ/kg"] = df_HHV["Gross Calorific Value-HHV (BTU/lb)"] * BtupLb_to_MJpKg

## composition 
desired_ultimate_analysis = "Ultimate Analysis (ASTM D 3176-09)"
# get composition
df_composition = df_fuelAnalyses[desired_ultimate_analysis]
# drop column "Notes Ultimate Analysis"
df_composition = df_composition.drop(columns=["Notes Ultimate Analysis"])

## ash
desired_proximate_analysis = "Proximate Analysis (ASTM D 3172-07)"
# get ash composition
df_ash = df_fuelAnalyses[desired_proximate_analysis]
# drop column "Notes Ultimate Analysis"
df_ash = df_ash[["Ash (%, db)","Moisture (%)"]]

## concatenate info, comp, and HHV
df_joined = pd.concat([df_info_red, df_HHV[["HHV, MJ/kg"]], df_composition, df_ash], axis=1)

## we have to pick feedstocks that have both HHV and ultimate analysis entries
df_joined = df_joined.dropna(subset=["HHV, MJ/kg"])
df_joined = df_joined.dropna(subset=["Oxygen-by difference (%, db)"])
df_joined = df_joined.reset_index()

# check mass balance
elemental_columns = ['Carbon (%, db)', 'Oxygen-by difference (%, db)',
                     'Hydrogen (%, db)', 'Nitrogen (%, db)',
                     'Sulfur (%, db)', 'Ash (%, db)']
df_joined[elemental_columns]

## if below detection limit (BDL), set to zero
df_joined['Nitrogen (%, db)'] = pd.to_numeric(df_joined['Nitrogen (%, db)'], errors='coerce').fillna(0)

## adjust ultimate composition to be dry and ash-free (daf), not just dry-basis (db)
ash_col = 'Ash (%, db)'  # adjust to your actual column name

# elemental_columns_daf = []
#
for col in elemental_columns[:-1]:
    daf_col = col.replace('db', 'daf')
    df_joined[daf_col] = df_joined[col] / (1 - df_joined[ash_col] / 100)






##################################################
### Phyllis data
# source: https://phyllis.nl/Browse/Standard/ECN-Phyllis
phyllis_folder_name = "Inputs/Phyllis2/"
file_names = [f for f in os.listdir(phyllis_folder_name) if f.startswith("Phyllis2 ")]

# determine relevant entries
relevantEntries = ["Material","ID-number","Country","Submission date",
                   "Moisture content",#"Ash content",
                   "Carbon", "Hydrogen", "Oxygen", "Nitrogen", "Sulphur",
                   "Gross calorific value (HHV)"]

# determine target column of relevant entries
targetCol_map = {}
targetCol_map["Material"] = 0
targetCol_map["ID-number"] = 0
targetCol_map["Country"] = 0
targetCol_map["Submission date"] = 0
targetCol_map["Moisture content"] = 1
targetCol_map["Ash content"] = 2
targetCol_map["Ash content at 550°C"] = 2
targetCol_map["Ash content at 815°C"] = 2

# where to save dfs for individual checking
df_phyllis = pd.DataFrame(columns=relevantEntries)
# stored which ones not saved
notStored = []
counter = 0
# read
for file_name in file_names:

    # if counter != 0:
    #     continue
    # read and limit to first 5 cols
    df_entries = pd.read_excel(phyllis_folder_name+file_name,
                               usecols=range(5))

    # special treatment for ash content
    mask = df_entries.iloc[:, 0].isin(relevantEntries) | df_entries.iloc[:, 0].str.startswith("Ash content", na=False)
    df_entries = df_entries[mask]

    # now we check; if there is no hydrogen then we move on
    if "Carbon" in df_entries.iloc[:, 0].values:
        # do nothing
        print("")
    else:
        notStored.append(file_name)
        continue

    # if there is 2 ash content, pick the first

    # rotate!
    df_entries = df_entries.T
    df_entries.columns = df_entries.iloc[0]  # use first row as header
    df_entries = df_entries.iloc[1:]          # drop the old header row
    df_entries = df_entries.loc[:, ~df_entries.columns.duplicated()]
    df_entries = df_entries.reset_index(drop=True)


    # collapse to single row by choosing one
    default_row = 3
    single_row = {}
    for col in df_entries.columns:
        try:
            row_idx = targetCol_map[col]
            val = df_entries[col].iloc[row_idx]
        except:
            val = df_entries[col].iloc[-1]  # fallback to last row
        single_row[col] = val

    # make result
    result = pd.DataFrame([single_row])
    # concatenate
    df_phyllis = pd.concat([df_phyllis, result], ignore_index=True).fillna(0)

    # counter
    counter += 1


## fix a bit ones without calculated oxygen
mask = (
    (df_phyllis["Oxygen"] == 0) &
    (df_phyllis["Carbon"] + df_phyllis["Hydrogen"] + df_phyllis["Sulphur"] + df_phyllis["Nitrogen"] < 80)
)

df_phyllis.loc[mask, "Oxygen"] = (
    100 - df_phyllis.loc[mask, "Carbon"]
    - df_phyllis.loc[mask, "Hydrogen"]
    - df_phyllis.loc[mask, "Sulphur"]
    - df_phyllis.loc[mask, "Nitrogen"]
)

## remove ones without HHV as dry and ash-free (daf)
df_phyllis = df_phyllis[df_phyllis["Gross calorific value (HHV)"] != 0].reset_index(drop=True)

# process further
ash_cols = [c for c in df_phyllis.columns if c.startswith("Ash content")]
#
df_phyllis["Ash content"] = df_phyllis[ash_cols].apply(
    lambda row: next((v for v in row if v != 0), 0), axis=1
)
#
df_phyllis = df_phyllis.drop(columns=[c for c in ash_cols if c != "Ash content"])

## get ones with moisture data
df_phyllis_moisture = df_phyllis[df_phyllis["Moisture content"] != 0].reset_index(drop=True)

#### DECLARE: in some cases there are no sulfur entries so it is zero





####################################################################################
### COMBINE PHYLLIS AND BFL DATA
# make copies
df_BT = df_joined.copy()
df_PH = df_phyllis_moisture.copy()

## rename BT
df_BT = df_BT.rename(columns={'Carbon (%, daf)': 'Carbon',
                              'Hydrogen (%, daf)': 'Hydrogen',
                              'Oxygen-by difference (%, daf)': 'Oxygen',
                              'Sulfur (%, daf)': 'Sulfur',
                              'Nitrogen (%, daf)': 'Nitrogen',
                              'Ash (%, db)': 'Ash content',
                              'Moisture (%)': 'Moisture content',
                              })
# add country
df_BT["Country"] = "USA"
df_BT["Dataset"] = "BFL"

## rename PH
df_PH = df_PH.rename(columns={'Gross calorific value (HHV)': 'HHV, MJ/kg',
                              'Material': 'CropType',
                              'Sulphur': 'Sulfur'})

# add data type
df_PH["Dataset"] = "Phyllis2"

## create combined data
df_combinedData = pd.concat([df_BT, df_PH], join='inner', ignore_index=True)


### OUTPUT
df_combinedData.to_csv("Outputs/biomassFeedstockData.csv", index=False)