from mof import MOF_CGCNN
import csv
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from pathlib import Path

def create_single_label_df(molecule, target_pressure):
    data_all_labels = pd.read_csv(f'{molecule}_data_all_labels.csv')
    data_all_labels.drop('surface_area_m2g', axis=1, inplace=True)

    target_pressure = f'{target_pressure}'
    columns_to_keep = ['id', 'surface_area_m2cm3', 'void_fraction', 'lcd', 'pld', target_pressure]
    data_single_label = data_all_labels.copy()
    data_single_label = data_single_label[columns_to_keep]
    data_single_label.rename(columns={target_pressure: 'target'}, inplace=True)

    # Reorder the columns as 'id', 'surface_area_m2_cm3', 'void_fraction', 'lcd', 'pld', 'target'
    column_order = ['id', 'surface_area_m2cm3', 'void_fraction', 'lcd', 'pld', 'target']
    data_single_label = data_single_label.reindex(columns=column_order)
    data_single_label = data_single_label.set_index('id')
    return data_single_label

def find_directory():
    """Finds the directory of the python script or Jupyter notebook.

    Returns:
        directory (str): directory of script.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # If running in a Jupter notebook
            directory = os.getcwd()
        else:
            directory = Path(__file__).parent
    except NameError:
        directory = Path(__file__).parent
    return directory

def get_cif_IDs():
    directory = find_directory()
    cif_directory = f'{directory}/cif'
    filenames = os.listdir(cif_directory)

    # Remove file extensions and return list
    cif_ids = [Path(file).stem for file in filenames if os.path.isfile(os.path.join(cif_directory, file))]
    return cif_ids

def build_dataset(molecule, target_pressure):
    # Make dataframe with single label at a given pressure
    data_single_label = create_single_label_df(molecule, target_pressure)

    # Filter dataframe so that it only contains MOFs that have corresponding cif files
    cif_ids = get_cif_IDs()
    data_single_label = data_single_label[data_single_label.index.isin(cif_ids)]

    # Save csv as training+validation set
    data_single_label.to_csv('data.csv', header=False)
    return data_single_label

def split_dataset(molecule, target_pressure, train_val_test_split):
    data_single_label = build_dataset(molecule=molecule, target_pressure=target_pressure)

    # Split away the test set
    training_val_set, test_set = train_test_split(data_single_label, test_size=train_val_test_split[2], random_state=42)

    # Save these dataframes as .csv files
    training_val_set.to_csv('training_val.csv', header=False)
    test_set.to_csv('test.csv', header=False)

# --- Hyperparameters ---
molecule = 'co2'
target_pressure = 0.1
train_val_test_split = [0.7, 0.2, 0.1]
epochs = 100
# -----------------------
    
split_dataset(molecule, target_pressure, train_val_test_split)

##read data
with open('./training_val.csv') as f:
    readerv = csv.reader(f)
    trainandval = [row for row in readerv]
with open('./test.csv') as f:
    readerv = csv.reader(f)
    test = [row for row in readerv]

test_size = train_val_test_split[1] / (1 - train_val_test_split[2])
train, val = train_test_split(trainandval, test_size=test_size, random_state=24)
#file path
root = './cif'
#create a class
mof = MOF_CGCNN(cuda=True,root_file=root,trainset = train, valset=val,testset=test,epoch = epochs,lr=0.002,optim='Adam',batch_size=24,h_fea_len=480,n_conv=5,lr_milestones=[200],weight_decay=1e-7,dropout=0.2)
# train the model
mof.train_MOF()

