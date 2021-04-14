#################### RDF Class ##################################
import os
import numpy as np
import pandas as pd

class rdf_reader:
    def __init__(self, std=None, mu=None, batch_size = 1, normalize=False, List_of_files="RDFs_Processed_Train.txt"):
        """Initializes a data reader for rdf of each MD frame, which gives back scaled or unscaled rdfs.
        Args:
            List_of_files(string): Name of text file storing data
            batch_size(int): Batch size
            RDF_Max(float): Maximum RDF in the data set to normalize
        """
        self.batchsize = batch_size
        self.data = np.array([])
        self.LoF = []
        with open(List_of_files) as f:
                self.LoF = f.read().split('\n')[:-1]
                f.close()
        self.n = len(self.LoF)
        self.num = 0
        self.normalize=normalize
        self.mu = mu
        self.std =std

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.num < self.n:
            self.data = np.array([])
            for i in range(self.batchsize):
                file = self.LoF[self.num]
                data_temp = pd.read_pickle(file)
                if  self.data.shape[0]==0:
                    data_temp = np.array(data_temp)
                    if self.normalize:
                        data_temp = (data_temp - self.mu)/(25.0*self.std**0.5 +1E-2)
                    self.data = data_temp    
                    self.data = self.data[:, np.newaxis]
                else:
                    data_temp = np.array(data_temp)
                    if self.normalize :
                        data_temp = (data_temp - self.mu)/(25.0*self.std**0.5 +1E-2)
                    self.data = np.append(self.data, data_temp[:,np.newaxis], axis=1)

                self.num = self.num + 1

                if self.num  >= self.n:
                    self.num = 0
            return self.data, self.data.shape

#################### Temperature Class ##################################

class temp_reader:
    def __init__(self,  quant_temp_Max, quant_temp_min, batch_size = 1,  List_of_files="Temps_Processed_Train.txt"):
        """Initializes a data reader for temperature of each MD simulation, which gives back scaled temperature.
        Args:
            List_of_files(string): Name of text file storing data
            batch_size(int): Batch size
            RDF_Max(float): Maximum RDF in the data set to normalize
        """

        self.batchsize = batch_size
        self.data_frame = pd.DataFrame()
        self.LoF = []
        with open(List_of_files) as f:
            self.LoF = f.read().split('\n')[:-1]
            f.close()
        self.n = len(self.LoF)
        self.num = 0
        for k, v_Max in quant_temp_Max.items():
            v_min = quant_temp_min[k]
            range_p = v_Max - v_min
        self.m = 1.0 / range_p
        self.b = - v_min /range_p


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.num < self.n:
            self.data_frame = pd.DataFrame()
            for i in range(self.batchsize):
                file = self.LoF[self.num]

                data_den= pd.read_csv(file, header = None,parse_dates=True,sep=' ')
                self.data_frame = self.data_frame.append(data_den)

                self.num = self.num + 1
                if self.num  >= self.n:
                # Reset batch iterator to start from first data
                    self.num = 0
            data = np.array(self.data_frame)
            data = data[:,-1]
            data = self.m * data + self.b

            return data, data.shape

#################### Density Class ##################################

class dens_reader:
    def __init__(self,quant_dens_Max, quant_dens_min, batch_size = 1,  List_of_files="Dens_Processed_Train.txt"):
        """Initializes a data reader for density of each MD simulation, which gives back scaled density.
        Args:
            List_of_files(string): Name of text file storing data
            batch_size(int): Batch size
            RDF_Max(float): Maximum RDF in the data set to normalize
        """
        self.batchsize = batch_size
        self.data_frame = pd.DataFrame()
        self.LoF = []
        with open(List_of_files) as f:
                self.LoF = f.read().split('\n')[:-1]
                f.close()
        self.n = len(self.LoF)
        self.num = 0
        for k, v_Max in quant_dens_Max.items():
            v_Max = (1/(v_Max**3))
            v_min = 1/quant_dens_min[k]**3
            range_p = v_min - v_Max
        self.m = 1.0 / range_p
        self.b = - v_Max /range_p
        
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.num < self.n:
            self.data_frame = pd.DataFrame()
            
            for i in range(self.batchsize):
                file = self.LoF[self.num]
                data_den= pd.read_csv(file, header = None,parse_dates=True,sep='\t')
                self.data_frame = self.data_frame.append(data_den)

                self.num = self.num + 1
                if self.num  >= self.n:
                # Reset batch iterator to start from first data
                    self.num = 0
            data = np.array(self.data_frame)
            data = self.m * 1.0/(data**3) + self.b

            return data, data.shape


##################### Combine Features and Convert #########################
def combine3(df1, df2,df3):
    np_df1 = np.array(df1)
    n_sample = np_df1.shape[0]
    n_data  = np_df1.shape[1]
    np_df2 = np.array(df2)
    np_df2 = np_df2.reshape((1,n_data, -1))
    np_df2 = np.repeat(np_df2[...],n_sample,0)
    np_df3 = np.array(df3)
    np_df3 = np_df3.reshape((1,n_data, -1))
    np_df3 = np.repeat(np_df3[...],n_sample,0)
    np_final = np.append(np_df1, np_df2, axis=-1)
    np_final = np.append(np_final, np_df3, axis=-1)

    return np_final
    
######## Function to be moved to Model or Data_Too Class ##################
################ Obtain Range for Each Feature #############################
def get_density_range(file_path, file_name='Den.csv'):
    folder_csv = os.path.join(file_path, file_name)
    Row_data = pd.read_csv(folder_csv, header =0 )
    Size_Max = Row_data['Density_Max'].astype(float).values
    Size_min = Row_data['Density_min'].astype(float).values

    quant_density_Max = {'Density': Size_Max}
    quant_density_min = {'Density': Size_min}

    return quant_density_Max, quant_density_min

def get_temperature_range(file_path, file_name='Temp.csv'):
    folder_csv = os.path.join(file_path, file_name)
    Row_data = pd.read_csv(folder_csv, header =0 )
    Temp_Max = Row_data['Temp_Max'].astype(float).values
    Temp_min = Row_data['Temp_min'].astype(float).values

    quant_temperature_Max = {'Temperature': Temp_Max}
    quant_temperature_min = {'Temperature': Temp_min}

    return quant_temperature_Max, quant_temperature_min