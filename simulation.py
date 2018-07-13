import os, fnmatch
import re
import math
from collections import defaultdict

import pandas as pd
import numpy as np
import random
from numpy import trapz
from scipy import stats, interpolate
from io import StringIO

import Orange


class Simulation:
    """
    Process tasks from client
    """
    def __init__(self):
        self.current_directory = os.getcwd()
        self.start_stringEd = r"Irradiances \(units of W\/m\^2 nm\)\, Mean Cosines \(Mubars\)\, and Irradiance " \
                              r"Reflectance at"
        self.stop_stringEd = r"LAYER\-AVERAGE K\-functions \(units of 1\/meter\)"

        self.start_stringKd = r"LAYER\-AVERAGE K\-functions \(units of 1\/meter\) at"
        self.stop_stringKd = r"Selected Radiances \(units of W\/m\^2 sr nm\) and Radiance\-Irradiance Ratios at"

        self.start_stringLwRrs = r"Selected Radiances \(units of W\/m\^2 sr nm\) and Radiance\-Irradiance Ratios at"
        self.stop_stringLwRrs = r"Waveband"

        self.df_kd_final = None
        self.df_kd_final_sensor = None

        # values aprox by the lowest or highest closer
        '''
        self.color_red_dictionary = {
            "360.0": 0.004,
            "380.0": 0.008,
            "400.0": 0.018,
            "420.0": 0.108,
            "440.0": 0.027,
            "460.0": 0.008,
            "480.0": 0.007,
            "500.0": 0.016,
            "520.0": 0.051,
            "540.0": 0.087,
            "560.0": 0.107,
            "580.0": 0.448,
            "600.0": 0.758,
            "620.0": 0.810,
            "640.0": 0.252,
            "660.0": 0.201,
            "680.0": 0.057,
            "700.0": 0.046,
            "720.0": 0.063,
            "740.0": 0.068,
            "760.0": 0.113,
            "780.0": 0.096,
            "795.0": 0.071
        }
        self.color_blue_dictionary = {
            "360.0": 0.005,
            "380.0": 0.008,
            "400.0": 0.065,
            "420.0": 0.425,
            "440.0": 0.500,
            "460.0": 0.555,
            "480.0": 0.520,
            "500.0": 0.370,
            "520.0": 0.198,
            "540.0": 0.127,
            "560.0": 0.086,
            "580.0": 0.091,
            "600.0": 0.092,
            "620.0": 0.085,
            "640.0": 0.042,
            "660.0": 0.030,
            "680.0": 0.015,
            "700.0": 0.016,
            "720.0": 0.017,
            "740.0": 0.012,
            "760.0": 0.039,
            "780.0": 0.052,
            "795.0": 0.065
        }
        self.color_green_dictionary = {
            "360.0": 0.004,
            "380.0": 0.004,
            "400.0": 0.010,
            "420.0": 0.048,
            "440.0": 0.082,
            "460.0": 0.122,
            "480.0": 0.265,
            "500.0": 0.339,
            "520.0": 0.565,
            "540.0": 0.639,
            "560.0": 0.553,
            "580.0": 0.315,
            "600.0": 0.108,
            "620.0": 0.060,
            "640.0": 0.262,
            "660.0": 0.020,
            "680.0": 0.008,
            "700.0": 0.010,
            "720.0": 0.023,
            "740.0": 0.026,
            "760.0": 0.054,
            "780.0": 0.058,
            "795.0": 0.047
        }
        '''
        # values from R, G, B .csv. Interpolated and normalized values
        self.color_red_dictionary = {}
        self.color_blue_dictionary = {}
        self.color_green_dictionary = {}
        self.clear_dictionary = {}

    def set_index(self, index):
        """
        Set index to dataframe
        :param index: it could be a list of names or pathnames
        :return:
        """
        self.df_kd_final = pd.DataFrame({'name': index})
        self.df_kd_final.set_index('name', inplace=True)

    def set_index_sensor(self, index):
        """
        Set index to sensor dataframe
        :param index: it could be a list of names or pathnames
        :return:
        """
        self.df_kd_final_sensor = pd.DataFrame({'name': index})
        self.df_kd_final_sensor.set_index('name', inplace=True)

    def process_output_file(self, content):
        # Select the useful info. Obtain Ed, Lw and Rrs from every wavelength

        # Obtain Ed
        patron = re.compile(r'{}\s*(?P<length>\d+\.\d+)\s*nm\s*(?P<table>[\s\S]*?){}'.format(self.start_stringEd,
                                                                                             self.stop_stringEd))
        selected_info = ""
        wavelength = ""

        # Creation of pandas dataframe with useful data
        df_final = pd.DataFrame()
        output_list = []

        # Regular expression to find the patron
        for m in re.finditer(patron, content):
            selected_info = m.group('table')
            wavelength = m.group('length')

            # Delete duplicated spaces and replace strings we need
            selected_info = re.sub(' +', ' ', selected_info)
            selected_info = re.sub('in air', '0 in_air in_air', selected_info)
            selected_info = re.sub('R = Eu/Ed', 'R=Eu/Ed', selected_info)

            data = StringIO(selected_info)

            # create dataframe for this patron iteration
            df = pd.read_csv(data, skipinitialspace=True, delimiter=' ')
            df.columns = ['iz', 'zeta', 'z(m)', 'Eou', 'Eod', 'Eo', 'Eu', 'Ed', 'MUBARu', 'MUBARd', 'MUBAR',
                          'R=Eu/Ed']

            # Replace values in df
            df['zeta'].replace('in_air', np.nan, inplace=True)
            df['z(m)'].replace('in_air', np.nan, inplace=True)

            # Delete unused rows
            df.drop(index=len(df) - 1, inplace=True)

            # Filtering data that we need
            df = df[['Ed', 'z(m)']]
            df['wavelength'] = wavelength
            df['calculated_Kd'] = 0
            df['calculated_intercept'] = 0
            df_final["Lw"] = 0
            df_final["Rrs"] = 0
            df_final["zupper"] = 0
            df_final["zlower"] = 0
            df_final["zmid"] = 0
            df_final["Kd"] = 0

            df = df.apply(pd.to_numeric, args=('coerce',))

            # Calculate df linear regression
            x = []
            y = []
            for i in range(0, len(df)):
                # if Ed is NaN in some depth, we take the n-1 Ed value for this Ed
                if math.isnan(df['Ed'].iloc[i]):
                    df['Ed'].iloc[i] = df['Ed'].iloc[i - 1]
                if not math.isnan(df['z(m)'].iloc[i]):
                    x.append(df['z(m)'].iloc[i])
                    y.append(math.log(df['Ed'].iloc[i]))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            df['calculated_Kd'] = slope * (-1)
            df['calculated_intercept'] = intercept

            df_final = df_final.append(df)

        # Obtain Kd
        patron = re.compile(r'{}\s*(?P<length>\d+\.\d+)\s*nm\s*(?P<table>[\s\S]*?){}'.format(self.start_stringKd,
                                                                                             self.stop_stringKd))
        # Regular expression to find the patron
        for m in re.finditer(patron, content):
            selected_info = m.group('table')
            wavelength = m.group('length')

            # Delete duplicated spaces
            selected_info = re.sub(' +', ' ', selected_info)

            data = StringIO(selected_info)

            # create dataframe for this patron iteration
            df = pd.read_csv(data, skipinitialspace=True, delimiter=' ')
            df.columns = ['zupper', 'zlower', 'zmid', 'Kou(z)', 'Kod(z)', 'Ko(z)', 'Ku(z)', 'Kd(z)',
                          'Knet(z)', 'KLu(z)']

            # Delete unused rows
            df.drop(index=len(df) - 1, inplace=True)
            df = df.apply(pd.to_numeric, args=('coerce',))

            # locate and add values from df to df_final of zupper, zlower, zmid and Kd(z)
            df_final.loc[(df_final['wavelength'] == float(wavelength)), "zupper"] = df['zupper']
            df_final.loc[(df_final['wavelength'] == float(wavelength)), "zlower"] = df['zlower']
            df_final.loc[(df_final['wavelength'] == float(wavelength)), "zmid"] = df['zmid']
            df_final.loc[(df_final['wavelength'] == float(wavelength)), "Kd"] = df['Kd(z)']

        # Obtain Lw and Rrs
        patron = re.compile(r'{}\s*(?P<length>\d+\.\d+)\s*nm\s*(?P<table>[\s\S]*?){}'.format(self.start_stringLwRrs,
                                                                                             self.stop_stringLwRrs))
        # Regular expression to find the patron
        for m in re.finditer(patron, content):
            selected_info = m.group('table')
            wavelength = m.group('length')

            # Delete duplicated spaces
            selected_info = re.sub(' +', ' ', selected_info)
            selected_info = re.sub('in air', '0 in_air in_air', selected_info)
            selected_info = re.sub('Q = Eu/Lu', 'Q=Eu/Lu', selected_info)
            selected_info = re.sub('Rrs = Lw/Ed', 'Rrs=Lw/Ed', selected_info)

            data = StringIO(selected_info)

            # create dataframe for this patron iteration
            df = pd.read_csv(data, skipinitialspace=True, delimiter=' ')
            df.columns = ['iz', 'zeta', 'z(m)', 'Lu(z)', 'Ld(z)', 'Lu/Ed', 'Q=Eu/Lu', 'Lw(z)', 'Rrs=Lw/Ed']

            # Replace values in df
            df['zeta'].replace('in_air', np.nan, inplace=True)
            df['z(m)'].replace('in_air', np.nan, inplace=True)

            # Delete unused rows
            df.drop(index=len(df) - 1, inplace=True)
            df.drop(index=0, inplace=True)
            df = df.apply(pd.to_numeric, args=('coerce',))

            # locate and add values from df to df_final of Lw and Rrs
            df_final.loc[(df_final['z(m)'].isnull()) & (df_final['wavelength'] == float(wavelength)), "Lw"] = \
                df['Lw(z)'].iloc[0]
            df_final.loc[(df_final['z(m)'].isnull()) & (df_final['wavelength'] == float(wavelength)), "Rrs"] = \
                df['Rrs=Lw/Ed'].iloc[0]

        # create JSON formatted as [{"wavelength": x, "values": [{"Ed": x, "Kd": x, "Lw": x, "Rrs": x, "depth": x,
        # "intercept": x},{...}]},{...}]
        df_final.replace(np.nan, "NaN", inplace=True)
        j = 0
        k = 0
        for i in (df_final['wavelength'].unique()):
            output = {}
            values_list = []
            values_list_Kd = []
            output["wavelength"] = i
            for z in (df_final['z(m)'].unique()):
                value = {}

                value["depth"] = z
                value["Ed"] = df_final['Ed'].iloc[j]
                value["calculated_Kd"] = df_final['calculated_Kd'].iloc[j]
                value["calculated_intercept"] = df_final['calculated_intercept'].iloc[j]
                value["Rrs"] = df_final['Rrs'].iloc[j]
                value["Lw"] = df_final['Lw'].iloc[j]
                values_list.append(value)
                j += 1
            output["values"] = values_list
            # output_list.append(output)

            for z in (df_final['z(m)'].unique()):
                value = {}

                value["zupper"] = df_final['zupper'].iloc[k]
                value["zlower"] = df_final['zlower'].iloc[k]
                value["zmid"] = df_final['zmid'].iloc[k]
                value["Kd"] = df_final['Kd'].iloc[k]

                values_list_Kd.append(value)
                k += 1
            output["values_Kd"] = values_list_Kd
            output_list.append(output)

        return output_list

    def process_all_output_file(self, pathname, content, file_name):
        """
        Find Ed in content for each depth and wavelength. Obtain Kd and save it in .csv
        :param pathname: name or pathname of each simulation
        :param content: output file of each simulation
        :return:
        """
        # Obtain Ed
        patron = re.compile(r'{}\s*(?P<length>\d+\.\d+)\s*nm\s*(?P<table>[\s\S]*?){}'.format(self.start_stringEd,
                                                                                             self.stop_stringEd))
        selected_info = ""
        wavelength = ""
        output_list = []

        # Regular expression to find the patron
        for m in re.finditer(patron, content):
            selected_info = m.group('table')
            wavelength = m.group('length')

            # Delete duplicated spaces and replace strings we need
            selected_info = re.sub(' +', ' ', selected_info)
            selected_info = re.sub('in air', '0 in_air in_air', selected_info)
            selected_info = re.sub('R = Eu/Ed', 'R=Eu/Ed', selected_info)

            data = StringIO(selected_info)

            # create dataframe for this patron iteration and obtain slope. Finally we add this slope to df_kd_final
            df = pd.read_csv(data, skipinitialspace=True, delimiter=' ')
            df.columns = ['iz', 'zeta', 'z(m)', 'Eou', 'Eod', 'Eo', 'Eu', 'Ed', 'MUBARu', 'MUBARd', 'MUBAR', 'R=Eu/Ed']

            # Replace values in df
            df['zeta'].replace('in_air', np.nan, inplace=True)
            df['z(m)'].replace('in_air', np.nan, inplace=True)

            # Delete unused rows
            df.drop(index=len(df)-1, inplace=True)

            # Filtering data that we need
            df = df[['Ed', 'z(m)']]
            df['wavelength'] = wavelength
            df['calculated_Kd'] = 0

            df = df.apply(pd.to_numeric, args=('coerce',))

            # Calculate df linear regression
            x = []
            y = []
            for i in range(0, len(df)):
                # if Ed is NaN in some depth, we take the n-1 Ed value for this Ed
                if math.isnan(df['Ed'].iloc[i]):
                    df['Ed'].iloc[i] = df['Ed'].iloc[i-1]
                if not math.isnan(df['z(m)'].iloc[i]):
                    x.append(df['z(m)'].iloc[i])
                    y.append(math.log(df['Ed'].iloc[i]))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            self.df_kd_final.loc[pathname, wavelength] = slope * (-1)

        self.df_kd_final.to_csv("distances/" + file_name + "_all_wavelengths.csv")

    def process_sensor_output_file(self, pathname, content, file_name):
        """
        Filter all the Ed results in Output file by wavelength.
        Normalize this values to red, green and blue values from sensor.
        Save in dataframe Kd results.
        :param pathname: name of the index in dataframe
        :param content: data in output file
        :return:
        """
        self.df_kd_final_sensor.loc[pathname, "RED"] = 0
        self.df_kd_final_sensor.loc[pathname, "GREEN"] = 0
        self.df_kd_final_sensor.loc[pathname, "BLUE"] = 0
        self.df_kd_final_sensor.loc[pathname, "CLEAR"] = 0

        selected_info = ""
        wavelength = ""
        count = 0
        output_list = []
        # create dict with values, and x (depth) and y (area) points for linear regression
        x_red = []
        y_red = []
        d_red = defaultdict(list)
        x_green = []
        y_green = []
        d_green = defaultdict(list)
        x_blue = []
        y_blue = []
        d_blue = defaultdict(list)
        x_clear = []
        y_clear = []
        d_clear = defaultdict(list)

        #normalize RGB values
        self.create_RGB_wavelength()


        # Obtain Ed
        patron = re.compile(r'{}\s*(?P<length>\d+\.\d+)\s*nm\s*(?P<table>[\s\S]*?){}'.format(self.start_stringEd,
                                                                                             self.stop_stringEd))

        # Regular expression to find the patron
        for m in re.finditer(patron, content):
            count += 1
            selected_info = m.group('table')
            wavelength = m.group('length')

            # Delete duplicated spaces and replace strings we need
            selected_info = re.sub(' +', ' ', selected_info)
            selected_info = re.sub('in air', '0 in_air in_air', selected_info)
            selected_info = re.sub('R = Eu/Ed', 'R=Eu/Ed', selected_info)

            data = StringIO(selected_info)

            # create dataframe for this patron iteration and obtain slope. Finally we add this slope to df_kd_final
            df = pd.read_csv(data, skipinitialspace=True, delimiter=' ')
            df.columns = ['iz', 'zeta', 'z(m)', 'Eou', 'Eod', 'Eo', 'Eu', 'Ed', 'MUBARu', 'MUBARd', 'MUBAR', 'R=Eu/Ed']

            # Replace values in df
            df['zeta'].replace('in_air', np.nan, inplace=True)
            df['z(m)'].replace('in_air', np.nan, inplace=True)

            # Delete unused rows
            df.drop(index=len(df)-1, inplace=True)

            # Filtering data that we need
            df = df[['Ed', 'z(m)']]
            df['wavelength'] = wavelength
            df['calculated_Kd'] = 0

            df = df.apply(pd.to_numeric, args=('coerce',))

            # save Ed for each depth
            if wavelength != "795.0":
                # Add Ed(z) * R or G or B values to dictionary
                for i in range(0, len(df)):
                    # if Ed is NaN in some depth, we take the n-1 Ed value for this Ed
                    if math.isnan(df['Ed'].iloc[i]):
                        df['Ed'].iloc[i] = df['Ed'].iloc[i - 1]
                    if not math.isnan(df['z(m)'].iloc[i]):
                        d_red[(df['z(m)'].iloc[i])].append((df['Ed'].iloc[i]) * self.color_red_dictionary[float(wavelength)])
                        d_green[(df['z(m)'].iloc[i])].append((df['Ed'].iloc[i]) * self.color_green_dictionary[float(wavelength)])
                        d_blue[(df['z(m)'].iloc[i])].append((df['Ed'].iloc[i]) * self.color_blue_dictionary[float(wavelength)])
                        d_clear[(df['z(m)'].iloc[i])].append((df['Ed'].iloc[i]) * self.clear_dictionary[float(wavelength)])

                        '''print("df['z(m)'].iloc[i]" + str(df['z(m)'].iloc[i]))
                        print("df['Ed'].iloc[i]" + str(df['Ed'].iloc[i]))
                        print("self.clear_dictionary[float(wavelength)]" + str(self.clear_dictionary[float(wavelength)]))
                        print("self.color_red_dictionary[float(wavelength)]" + str(self.color_red_dictionary[float(wavelength)]))
                        print("self.color_green_dictionary[float(wavelength)]" + str(self.color_blue_dictionary[float(wavelength)]))
                        print("self.color_blue_dictionary[float(wavelength)]" + str(self.color_green_dictionary[float(wavelength)]))
                        print("d_red[(df['z(m)'].iloc[i])]" + str(d_red[(df['z(m)'].iloc[i])]))
                        print("d_green[(df['z(m)'].iloc[i])]" + str(d_green[(df['z(m)'].iloc[i])]))
                        print("d_blue[(df['z(m)'].iloc[i])]" + str(d_blue[(df['z(m)'].iloc[i])]))
                        print("d_clear[(df['z(m)'].iloc[i])]" + str(d_clear[(df['z(m)'].iloc[i])]))
                        '''

        # integrate values for each depth with composite trapezoidal
        for key, value in d_red.items():
            #print("red")
            #print("key - depth" + str(key))
            #print("value" + str(value))
            y = np.array(value)
            #print("y - values of Ed*R" + str(y))
            area = trapz(y, dx=5)
            #print("area" + str(area))
            x_red.append(key)
            y_red.append(math.log(area))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_red, y_red)
        self.df_kd_final_sensor.loc[pathname, "RED"] = slope * (-1)
        #print("kd_red" + str(self.df_kd_final_sensor.loc[pathname, "RED"]))

        # integrate values for each depth with composite trapezoidal
        for key, value in d_green.items():
            y = np.array(value)
            area = trapz(y, dx=5)
            x_green.append(key)
            y_green.append(math.log(area))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_green, y_green)
        self.df_kd_final_sensor.loc[pathname, "GREEN"] = slope * (-1)
        #print("kd_green" + str(self.df_kd_final_sensor.loc[pathname, "GREEN"]))

        # integrate values for each depth with composite trapezoidal
        for key, value in d_blue.items():
            y = np.array(value)
            area = trapz(y, dx=5)
            x_blue.append(key)
            y_blue.append(math.log(area))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_blue, y_blue)
        self.df_kd_final_sensor.loc[pathname, "BLUE"] = slope * (-1)
        #print("kd_blue" + str(self.df_kd_final_sensor.loc[pathname, "BLUE"]))

        # integrate values for each depth with composite trapezoidal
        for key, value in d_clear.items():
            y = np.array(value)
            area = trapz(y, dx=5)
            x_clear.append(key)
            y_clear.append(math.log(area))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clear, y_clear)
        self.df_kd_final_sensor.loc[pathname, "CLEAR"] = slope * (-1)
        #print("kd_clear" + str(self.df_kd_final_sensor.loc[pathname, "CLEAR"]))

        self.df_kd_final_sensor.to_csv("distances/" + file_name + "_all_sensors.csv")

        return self.df_kd_final_sensor.loc[pathname, "RED"], self.df_kd_final_sensor.loc[pathname, "GREEN"], \
               self.df_kd_final_sensor.loc[pathname, "BLUE"], self.df_kd_final_sensor.loc[pathname, "CLEAR"]

    def process_sensor_output_file_without_dict(self, pathname, content, file_name):
        """
        Filter all the Ed results in Output file by wavelength.
        Normalize this values to red, green and blue values from sensor.
        Save in dataframe Kd results.
        :param pathname: name of the index in dataframe
        :param content: data in output file
        :return:
        """
        self.df_kd_final_sensor.loc[pathname, "RED"] = 0
        self.df_kd_final_sensor.loc[pathname, "GREEN"] = 0
        self.df_kd_final_sensor.loc[pathname, "BLUE"] = 0
        self.df_kd_final_sensor.loc[pathname, "CLEAR"] = 0

        selected_info = ""
        wavelength = ""
        count = 0
        output_list = []
        # create dict with values, and x (depth) and y (area) points for linear regression
        x_red = []
        y_red = []
        d_red = defaultdict(list)
        x_green = []
        y_green = []
        d_green = defaultdict(list)
        x_blue = []
        y_blue = []
        d_blue = defaultdict(list)
        x_clear = []
        y_clear = []
        d_clear = defaultdict(list)

        #normalize RGB values
        self.create_RGB_wavelength()


        # Obtain Ed
        patron = re.compile(r'{}\s*(?P<length>\d+\.\d+)\s*nm\s*(?P<table>[\s\S]*?){}'.format(self.start_stringEd,
                                                                                             self.stop_stringEd))

        # Regular expression to find the patron
        for m in re.finditer(patron, content):
            count += 1
            selected_info = m.group('table')
            wavelength = m.group('length')

            # Delete duplicated spaces and replace strings we need
            selected_info = re.sub(' +', ' ', selected_info)
            selected_info = re.sub('in air', '0 in_air in_air', selected_info)
            selected_info = re.sub('R = Eu/Ed', 'R=Eu/Ed', selected_info)

            data = StringIO(selected_info)

            # create dataframe for this patron iteration and obtain slope. Finally we add this slope to df_kd_final
            df = pd.read_csv(data, skipinitialspace=True, delimiter=' ')
            df.columns = ['iz', 'zeta', 'z(m)', 'Eou', 'Eod', 'Eo', 'Eu', 'Ed', 'MUBARu', 'MUBARd', 'MUBAR', 'R=Eu/Ed']

            # Replace values in df
            df['zeta'].replace('in_air', np.nan, inplace=True)
            df['z(m)'].replace('in_air', np.nan, inplace=True)

            # Delete unused rows
            df.drop(index=len(df)-1, inplace=True)

            # Filtering data that we need
            df = df[['Ed', 'z(m)']]
            df['wavelength'] = wavelength
            df['calculated_Kd'] = 0

            df = df.apply(pd.to_numeric, args=('coerce',))

            # save Ed for each depth
            if wavelength != "795.0":
                # Add Ed(z) * R or G or B values to dictionary
                for i in range(0, len(df)):
                    # if Ed is NaN in some depth, we take the n-1 Ed value for this Ed
                    if math.isnan(df['Ed'].iloc[i]):
                        df['Ed'].iloc[i] = df['Ed'].iloc[i - 1]
                    if not math.isnan(df['z(m)'].iloc[i]):
                        d_red[(df['z(m)'].iloc[i])].append((df['Ed'].iloc[i]))
                        d_green[(df['z(m)'].iloc[i])].append((df['Ed'].iloc[i]))
                        d_blue[(df['z(m)'].iloc[i])].append((df['Ed'].iloc[i]))
                        d_clear[(df['z(m)'].iloc[i])].append((df['Ed'].iloc[i]))

                        '''print("df['z(m)'].iloc[i]" + str(df['z(m)'].iloc[i]))
                        print("df['Ed'].iloc[i]" + str(df['Ed'].iloc[i]))
                        print("self.clear_dictionary[float(wavelength)]" + str(self.clear_dictionary[float(wavelength)]))
                        print("self.color_red_dictionary[float(wavelength)]" + str(self.color_red_dictionary[float(wavelength)]))
                        print("self.color_green_dictionary[float(wavelength)]" + str(self.color_blue_dictionary[float(wavelength)]))
                        print("self.color_blue_dictionary[float(wavelength)]" + str(self.color_green_dictionary[float(wavelength)]))
                        print("d_red[(df['z(m)'].iloc[i])]" + str(d_red[(df['z(m)'].iloc[i])]))
                        print("d_green[(df['z(m)'].iloc[i])]" + str(d_green[(df['z(m)'].iloc[i])]))
                        print("d_blue[(df['z(m)'].iloc[i])]" + str(d_blue[(df['z(m)'].iloc[i])]))
                        print("d_clear[(df['z(m)'].iloc[i])]" + str(d_clear[(df['z(m)'].iloc[i])]))
                        '''

        # integrate values for each depth with composite trapezoidal
        for key, value in d_red.items():
            #print("red")
            #print("key - depth" + str(key))
            #print("value" + str(value))
            y = np.array(value)
            #print("y - values of Ed*R" + str(y))
            area = trapz(y, dx=5)
            #print("area" + str(area))
            x_red.append(key)
            y_red.append(math.log(area))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_red, y_red)
        self.df_kd_final_sensor.loc[pathname, "RED"] = slope * (-1)
        #print("kd_red" + str(self.df_kd_final_sensor.loc[pathname, "RED"]))

        # integrate values for each depth with composite trapezoidal
        for key, value in d_green.items():
            y = np.array(value)
            area = trapz(y, dx=5)
            x_green.append(key)
            y_green.append(math.log(area))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_green, y_green)
        self.df_kd_final_sensor.loc[pathname, "GREEN"] = slope * (-1)
        #print("kd_green" + str(self.df_kd_final_sensor.loc[pathname, "GREEN"]))

        # integrate values for each depth with composite trapezoidal
        for key, value in d_blue.items():
            y = np.array(value)
            area = trapz(y, dx=5)
            x_blue.append(key)
            y_blue.append(math.log(area))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_blue, y_blue)
        self.df_kd_final_sensor.loc[pathname, "BLUE"] = slope * (-1)
        #print("kd_blue" + str(self.df_kd_final_sensor.loc[pathname, "BLUE"]))

        # integrate values for each depth with composite trapezoidal
        for key, value in d_clear.items():
            y = np.array(value)
            area = trapz(y, dx=5)
            x_clear.append(key)
            y_clear.append(math.log(area))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clear, y_clear)
        self.df_kd_final_sensor.loc[pathname, "CLEAR"] = slope * (-1)
        #print("kd_clear" + str(self.df_kd_final_sensor.loc[pathname, "CLEAR"]))

        self.df_kd_final_sensor.to_csv("distances/" + file_name + "_all_sensors_without_dict.csv")

        return self.df_kd_final_sensor.loc[pathname, "RED"], self.df_kd_final_sensor.loc[pathname, "GREEN"], \
               self.df_kd_final_sensor.loc[pathname, "BLUE"], self.df_kd_final_sensor.loc[pathname, "CLEAR"]

    def process_sensor_z_output_file(self, pathname, content, file_name):
        """
        Filter all the Ed results in Output file by wavelength.
        Normalize this values to red, green and blue values from sensor.
        Save in dataframe Kd results.
        :param pathname: name of the index in dataframe
        :param content: data in output file
        :return:
        """

        #noise = np.random.normal(1, 20, 100)
        #print(noise)
        self.df_kd_final_sensor.loc[pathname, "RED"] = 0
        self.df_kd_final_sensor.loc[pathname, "GREEN"] = 0
        self.df_kd_final_sensor.loc[pathname, "BLUE"] = 0
        self.df_kd_final_sensor.loc[pathname, "CLEAR"] = 0

        selected_info = ""
        wavelength = ""
        count = 0
        output_list = []
        # create dict with values, and x (depth) and y (area) points for linear regression
        x_red = []
        y_red = []
        d_red = defaultdict(list)
        x_green = []
        y_green = []
        d_green = defaultdict(list)
        x_blue = []
        y_blue = []
        d_blue = defaultdict(list)
        x_clear = []
        y_clear = []
        d_clear = defaultdict(list)

        #normalize RGB values
        self.create_RGB_wavelength()

        # Obtain Ed
        patron = re.compile(r'{}\s*(?P<length>\d+\.\d+)\s*nm\s*(?P<table>[\s\S]*?){}'.format(self.start_stringEd,
                                                                                             self.stop_stringEd))

        # Regular expression to find the patron
        for m in re.finditer(patron, content):
            count += 1
            selected_info = m.group('table')
            wavelength = m.group('length')

            # Delete duplicated spaces and replace strings we need
            selected_info = re.sub(' +', ' ', selected_info)
            selected_info = re.sub('in air', '0 in_air in_air', selected_info)
            selected_info = re.sub('R = Eu/Ed', 'R=Eu/Ed', selected_info)

            data = StringIO(selected_info)

            # create dataframe for this patron iteration and obtain slope. Finally we add this slope to df_kd_final
            df = pd.read_csv(data, skipinitialspace=True, delimiter=' ')
            df.columns = ['iz', 'zeta', 'z(m)', 'Eou', 'Eod', 'Eo', 'Eu', 'Ed', 'MUBARu', 'MUBARd', 'MUBAR', 'R=Eu/Ed']

            # Replace values in df
            df['zeta'].replace('in_air', np.nan, inplace=True)
            df['z(m)'].replace('in_air', np.nan, inplace=True)

            # Delete unused rows
            df.drop(index=len(df)-1, inplace=True)

            # Filtering data that we need
            df = df[['Ed', 'z(m)']]
            df['wavelength'] = wavelength
            df['calculated_Kd'] = 0

            df = df.apply(pd.to_numeric, args=('coerce',))

            # save Ed for each depth
            if wavelength != "795.0":
                # Add Ed(z) * R or G or B values to dictionary. Z will be: 0.3, 0.6, 0.9, 1.25 and 1.5
                for i in np.arange(0, len(df)):
                    if df['z(m)'].iloc[i] == 0.3:
                        d_red[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_red_dictionary[float(wavelength)])
                        d_green[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_green_dictionary[float(wavelength)])
                        d_blue[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_blue_dictionary[float(wavelength)])
                        d_clear[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.clear_dictionary[float(wavelength)])
                    if df['z(m)'].iloc[i] == 0.6:
                        d_red[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_red_dictionary[float(wavelength)])
                        d_green[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_green_dictionary[float(wavelength)])
                        d_blue[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_blue_dictionary[float(wavelength)])
                        d_clear[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.clear_dictionary[float(wavelength)])
                    if df['z(m)'].iloc[i] == 0.9:
                        d_red[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_red_dictionary[float(wavelength)])
                        d_green[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_green_dictionary[float(wavelength)])
                        d_blue[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_blue_dictionary[float(wavelength)])
                        d_clear[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.clear_dictionary[float(wavelength)])
                    if df['z(m)'].iloc[i] == 1.25:
                        d_red[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_red_dictionary[float(wavelength)])
                        d_green[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_green_dictionary[float(wavelength)])
                        d_blue[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_blue_dictionary[float(wavelength)])
                        d_clear[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.clear_dictionary[float(wavelength)])

        # integrate values for each depth with composite trapezoidal
        for key, value in d_red.items():
            y = np.array(value)
            area = trapz(y, dx=5)
            x_red.append(key)
            y_red.append(math.log(area))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_red, y_red)
        self.df_kd_final_sensor.loc[pathname, "RED"] = slope * (-1)
        #print("kd" + str(self.df_kd_final_sensor.loc[pathname, "RED"]))

        # integrate values for each depth with composite trapezoidal
        for key, value in d_green.items():
            y = np.array(value)
            area = trapz(y, dx=5)
            x_green.append(key)
            y_green.append(math.log(area))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_green, y_green)
        self.df_kd_final_sensor.loc[pathname, "GREEN"] = slope * (-1)
        #print("kd" + str(self.df_kd_final_sensor.loc[pathname, "GREEN"]))

        # integrate values for each depth with composite trapezoidal
        for key, value in d_blue.items():
            y = np.array(value)
            area = trapz(y, dx=5)
            x_blue.append(key)
            y_blue.append(math.log(area))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_blue, y_blue)
        self.df_kd_final_sensor.loc[pathname, "BLUE"] = slope * (-1)
        #print("kd" + str(self.df_kd_final_sensor.loc[pathname, "BLUE"]))

        # integrate values for each depth with composite trapezoidal
        for key, value in d_clear.items():
            y = np.array(value)
            area = trapz(y, dx=5)
            x_clear.append(key)
            y_clear.append(math.log(area))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clear, y_clear)
        self.df_kd_final_sensor.loc[pathname, "CLEAR"] = slope * (-1)
        #print("kd" + str(self.df_kd_final_sensor.loc[pathname, "CLEAR"]))

        self.df_kd_final_sensor.to_csv("distances/" + file_name + "_0.3_0.6_0.9_1.25_sensor.csv")

        return self.df_kd_final_sensor.loc[pathname, "RED"], self.df_kd_final_sensor.loc[pathname, "GREEN"], \
               self.df_kd_final_sensor.loc[pathname, "BLUE"], self.df_kd_final_sensor.loc[pathname, "CLEAR"]

    def process_sensor_z_error_output_file(self, pathname, content, error, file_name):
        """
        Filter all the Ed results in Output file by wavelength.
        Normalize this values to red, green and blue values from sensor.
        Save in dataframe Kd results.
        :param pathname: name of the index in dataframe
        :param content: data in output file
        :return:
        """

        #todo
        #noise = np.random.normal(0.0, 1.0, 5)
        #print(noise)

        self.df_kd_final_sensor.loc[pathname, "RED"] = 0
        self.df_kd_final_sensor.loc[pathname, "GREEN"] = 0
        self.df_kd_final_sensor.loc[pathname, "BLUE"] = 0
        self.df_kd_final_sensor.loc[pathname, "CLEAR"] = 0

        selected_info = ""
        wavelength = ""
        count = 0
        output_list = []
        # create dict with values, and x (depth) and y (area) points for linear regression
        x_red = []
        y_red = []
        d_red = defaultdict(list)
        x_green = []
        y_green = []
        d_green = defaultdict(list)
        x_blue = []
        y_blue = []
        d_blue = defaultdict(list)
        x_clear = []
        y_clear = []
        d_clear = defaultdict(list)

        #normalize RGB values
        self.create_RGB_wavelength()

        # Obtain Ed
        patron = re.compile(r'{}\s*(?P<length>\d+\.\d+)\s*nm\s*(?P<table>[\s\S]*?){}'.format(self.start_stringEd,
                                                                                             self.stop_stringEd))

        # Regular expression to find the patron
        for m in re.finditer(patron, content):
            count += 1
            selected_info = m.group('table')
            wavelength = m.group('length')

            # Delete duplicated spaces and replace strings we need
            selected_info = re.sub(' +', ' ', selected_info)
            selected_info = re.sub('in air', '0 in_air in_air', selected_info)
            selected_info = re.sub('R = Eu/Ed', 'R=Eu/Ed', selected_info)

            data = StringIO(selected_info)

            # create dataframe for this patron iteration and obtain slope. Finally we add this slope to df_kd_final
            df = pd.read_csv(data, skipinitialspace=True, delimiter=' ')
            df.columns = ['iz', 'zeta', 'z(m)', 'Eou', 'Eod', 'Eo', 'Eu', 'Ed', 'MUBARu', 'MUBARd', 'MUBAR', 'R=Eu/Ed']

            # Replace values in df
            df['zeta'].replace('in_air', np.nan, inplace=True)
            df['z(m)'].replace('in_air', np.nan, inplace=True)

            # Delete unused rows
            df.drop(index=len(df)-1, inplace=True)

            # Filtering data that we need
            df = df[['Ed', 'z(m)']]
            df['wavelength'] = wavelength
            df['calculated_Kd'] = 0

            df = df.apply(pd.to_numeric, args=('coerce',))

            # save Ed for each depth
            if wavelength != "795.0":
                # Add Ed(z) * R or G or B values to dictionary. Z will be: 0.3, 0.6, 0.9, 1.25 and 1.5
                for i in np.arange(0, len(df)):
                    if df['z(m)'].iloc[i] == 0.3:

                        d_red[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_red_dictionary[float(wavelength)])
                        d_green[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_green_dictionary[float(wavelength)])
                        d_blue[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_blue_dictionary[float(wavelength)])
                        d_clear[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.clear_dictionary[float(wavelength)])

                    if df['z(m)'].iloc[i] == 0.6:

                        d_red[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_red_dictionary[float(wavelength)])
                        d_green[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_green_dictionary[float(wavelength)])
                        d_blue[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_blue_dictionary[float(wavelength)])
                        d_clear[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.clear_dictionary[float(wavelength)])

                    if df['z(m)'].iloc[i] == 0.9:

                        d_red[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_red_dictionary[float(wavelength)])
                        d_green[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_green_dictionary[float(wavelength)])
                        d_blue[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_blue_dictionary[float(wavelength)])
                        d_clear[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.clear_dictionary[float(wavelength)])

                    if df['z(m)'].iloc[i] == 1.25:

                        d_red[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_red_dictionary[float(wavelength)])
                        d_green[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_green_dictionary[float(wavelength)])
                        d_blue[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_blue_dictionary[float(wavelength)])
                        d_clear[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.clear_dictionary[float(wavelength)])

        # integrate values for each depth with composite trapezoidal
        for key, value in d_red.items():
            y = np.array(value)
            area = trapz(y, dx=5)
            x_red.append(key)
            random_error = self.create_error(error)
            #print("random_error" + str(random_error))
            error_to_sum = self.calc_error(random_error,area)
            #print("error_to_sum" + str(error_to_sum))
            #print("area without error" + str(area))
            area = area + error_to_sum
            #print("area with error" + str(area))
            y_red.append(math.log(area))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_red, y_red)
        self.df_kd_final_sensor.loc[pathname, "RED"] = slope * (-1)
        #print("kd_red:" + str(self.df_kd_final_sensor.loc[pathname, "RED"]))

        # integrate values for each depth with composite trapezoidal
        for key, value in d_green.items():
            y = np.array(value)
            area = trapz(y, dx=5)
            x_green.append(key)
            random_error = self.create_error(error)
            #print("random_error" + str(random_error))
            error_to_sum = self.calc_error(random_error,area)
            #print("error_to_sum" + str(error_to_sum))
            #print("area without error" + str(area))
            area = area + error_to_sum
            #print("area with error" + str(area))
            y_green.append(math.log(area))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_green, y_green)
        self.df_kd_final_sensor.loc[pathname, "GREEN"] = slope * (-1)
        #print("kd" + str(self.df_kd_final_sensor.loc[pathname, "GREEN"]))

        # integrate values for each depth with composite trapezoidal
        for key, value in d_blue.items():
            y = np.array(value)
            area = trapz(y, dx=5)
            x_blue.append(key)
            random_error = self.create_error(error)
            #print("random_error" + str(random_error))
            error_to_sum = self.calc_error(random_error,area)
            #print("error_to_sum" + str(error_to_sum))
            #print("area without error" + str(area))
            area = area + error_to_sum
            #print("area with error" + str(area))
            y_blue.append(math.log(area))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_blue, y_blue)
        self.df_kd_final_sensor.loc[pathname, "BLUE"] = slope * (-1)
        #print("kd" + str(self.df_kd_final_sensor.loc[pathname, "BLUE"]))

        # integrate values for each depth with composite trapezoidal
        for key, value in d_clear.items():
            y = np.array(value)
            area = trapz(y, dx=5)
            x_clear.append(key)
            random_error = self.create_error(error)
            #print("random_error" + str(random_error))
            error_to_sum = self.calc_error(random_error,area)
            #print("error_to_sum" + str(error_to_sum))
            #print("area without error" + str(area))
            area = area + error_to_sum
            #print("area with error" + str(area))
            y_clear.append(math.log(area))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clear, y_clear)
        self.df_kd_final_sensor.loc[pathname, "CLEAR"] = slope * (-1)
        #print("kd" + str(self.df_kd_final_sensor.loc[pathname, "CLEAR"]))

        self.df_kd_final_sensor.to_csv("distances/" + file_name + "_0.3_0.6_0.9_1.25_sensor_error" + error + ".csv")

        return self.df_kd_final_sensor.loc[pathname, "RED"], self.df_kd_final_sensor.loc[pathname, "GREEN"], \
               self.df_kd_final_sensor.loc[pathname, "BLUE"], self.df_kd_final_sensor.loc[pathname, "CLEAR"]

    def process_sensor_double_z_error_output_file(self, pathname, content, error, file_name):
        """
        Filter all the Ed results in Output file by wavelength.
        Normalize this values to red, green and blue values from sensor.
        Save in dataframe Kd results.
        :param pathname: name of the index in dataframe
        :param content: data in output file
        :return:
        """

        #todo
        #noise = np.random.normal(0.0, 1.0, 5)
        #print(noise)


        self.df_kd_final_sensor.loc[pathname, "RED"] = 0
        self.df_kd_final_sensor.loc[pathname, "GREEN"] = 0
        self.df_kd_final_sensor.loc[pathname, "BLUE"] = 0
        self.df_kd_final_sensor.loc[pathname, "CLEAR"] = 0

        selected_info = ""
        wavelength = ""
        count = 0
        output_list = []
        # create dict with values, and x (depth) and y (area) points for linear regression
        x_red = []
        y_red = []
        d_red = defaultdict(list)
        x_green = []
        y_green = []
        d_green = defaultdict(list)
        x_blue = []
        y_blue = []
        d_blue = defaultdict(list)
        x_clear = []
        y_clear = []
        d_clear = defaultdict(list)

        #normalize RGB values
        self.create_RGB_wavelength()

        # Obtain Ed
        patron = re.compile(r'{}\s*(?P<length>\d+\.\d+)\s*nm\s*(?P<table>[\s\S]*?){}'.format(self.start_stringEd,
                                                                                             self.stop_stringEd))

        # Regular expression to find the patron
        for m in re.finditer(patron, content):
            count += 1
            selected_info = m.group('table')
            wavelength = m.group('length')

            # Delete duplicated spaces and replace strings we need
            selected_info = re.sub(' +', ' ', selected_info)
            selected_info = re.sub('in air', '0 in_air in_air', selected_info)
            selected_info = re.sub('R = Eu/Ed', 'R=Eu/Ed', selected_info)

            data = StringIO(selected_info)

            # create dataframe for this patron iteration and obtain slope. Finally we add this slope to df_kd_final
            df = pd.read_csv(data, skipinitialspace=True, delimiter=' ')
            df.columns = ['iz', 'zeta', 'z(m)', 'Eou', 'Eod', 'Eo', 'Eu', 'Ed', 'MUBARu', 'MUBARd', 'MUBAR', 'R=Eu/Ed']

            # Replace values in df
            df['zeta'].replace('in_air', np.nan, inplace=True)
            df['z(m)'].replace('in_air', np.nan, inplace=True)

            # Delete unused rows
            df.drop(index=len(df)-1, inplace=True)

            # Filtering data that we need
            df = df[['Ed', 'z(m)']]
            df['wavelength'] = wavelength
            df['calculated_Kd'] = 0

            df = df.apply(pd.to_numeric, args=('coerce',))

            # save Ed for each depth
            if wavelength != "795.0":
                # Add Ed(z) * R or G or B values to dictionary. Z will be: 0.3, 0.6, 0.9, 1.25 and 1.5
                for i in np.arange(0, len(df)):
                    if df['z(m)'].iloc[i] == 0.3:

                        d_red[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_red_dictionary[float(wavelength)])
                        d_green[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_green_dictionary[float(wavelength)])
                        d_blue[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_blue_dictionary[float(wavelength)])
                        d_clear[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.clear_dictionary[float(wavelength)])

                    if df['z(m)'].iloc[i] == 0.6:

                        d_red[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_red_dictionary[float(wavelength)])
                        d_green[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_green_dictionary[float(wavelength)])
                        d_blue[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_blue_dictionary[float(wavelength)])
                        d_clear[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.clear_dictionary[float(wavelength)])

                    if df['z(m)'].iloc[i] == 0.9:

                        d_red[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_red_dictionary[float(wavelength)])
                        d_green[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_green_dictionary[float(wavelength)])
                        d_blue[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_blue_dictionary[float(wavelength)])
                        d_clear[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.clear_dictionary[float(wavelength)])

                    if df['z(m)'].iloc[i] == 1.25:

                        d_red[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_red_dictionary[float(wavelength)])
                        d_green[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_green_dictionary[float(wavelength)])
                        d_blue[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.color_blue_dictionary[float(wavelength)])
                        d_clear[(df['z(m)'].iloc[i])].append(
                            (df['Ed'].iloc[i]) * self.clear_dictionary[float(wavelength)])

        # integrate values for each depth with composite trapezoidal
        # key = 0.3, 0.6, 0.9, 1.3
        # values = each wavelength normalized to RGBC spectra (89 wavelength)
        # take 2 samples for each key (depth). From the same Ed, calculate 2 errors (2 samples, and calculate its mean)
        # calculate linear regresion with x as depth (key) and y as log (area + error) of wavelength
        for key, value in d_red.items():

            #print("key " + str(key))
            #print("value " + str(value))
            #print("length values : " + str(len(value)))
            y = np.array(value)
            area = trapz(y, dx=5)
            x_red.append(key)
            random_error1 = self.create_error(error)
            random_error2 = self.create_error(error)
            #print("error1 :" + str(random_error1))
            #print("error2 :" + str(random_error2))
            random_error = (random_error1 + random_error2)/2
            #print("error mean :" + str(random_error))
            #print("random_error" + str(random_error))
            error_to_sum = self.calc_error(random_error,area)
            #print("error_to_sum" + str(error_to_sum))
            #print("area without error" + str(area))
            area = area + error_to_sum
            #print("area with error" + str(area))
            y_red.append(math.log(area))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_red, y_red)
        self.df_kd_final_sensor.loc[pathname, "RED"] = slope * (-1)
        #print("kd_red:" + str(self.df_kd_final_sensor.loc[pathname, "RED"]))

        # integrate values for each depth with composite trapezoidal
        for key, value in d_green.items():
            y = np.array(value)
            area = trapz(y, dx=5)
            x_green.append(key)
            random_error1 = self.create_error(error)
            random_error2 = self.create_error(error)
            #print("error1 :" + str(random_error1))
            #print("error2 :" + str(random_error2))
            random_error = (random_error1 + random_error2) / 2
            #print("random_error" + str(random_error))
            error_to_sum = self.calc_error(random_error,area)
            #print("error_to_sum" + str(error_to_sum))
            #print("area without error" + str(area))
            area = area + error_to_sum
            #print("area with error" + str(area))
            y_green.append(math.log(area))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_green, y_green)
        self.df_kd_final_sensor.loc[pathname, "GREEN"] = slope * (-1)
        #print("kd" + str(self.df_kd_final_sensor.loc[pathname, "GREEN"]))

        # integrate values for each depth with composite trapezoidal
        for key, value in d_blue.items():
            y = np.array(value)
            area = trapz(y, dx=5)
            x_blue.append(key)
            random_error1 = self.create_error(error)
            random_error2 = self.create_error(error)
            #print("error1 :" + str(random_error1))
            #print("error2 :" + str(random_error2))
            random_error = (random_error1 + random_error2) / 2
            #print("random_error" + str(random_error))
            error_to_sum = self.calc_error(random_error,area)
            #print("error_to_sum" + str(error_to_sum))
            #print("area without error" + str(area))
            area = area + error_to_sum
            #print("area with error" + str(area))
            y_blue.append(math.log(area))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_blue, y_blue)
        self.df_kd_final_sensor.loc[pathname, "BLUE"] = slope * (-1)
        #print("kd" + str(self.df_kd_final_sensor.loc[pathname, "BLUE"]))

        # integrate values for each depth with composite trapezoidal
        for key, value in d_clear.items():
            y = np.array(value)
            area = trapz(y, dx=5)
            x_clear.append(key)
            random_error1 = self.create_error(error)
            random_error2 = self.create_error(error)
            #print("error1 :" + str(random_error1))
            #print("error2 :" + str(random_error2))
            random_error = (random_error1 + random_error2) / 2
            #print("random_error" + str(random_error))
            error_to_sum = self.calc_error(random_error,area)
            #print("error_to_sum" + str(error_to_sum))
            #print("area without error" + str(area))
            area = area + error_to_sum
            #print("area with error" + str(area))
            y_clear.append(math.log(area))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clear, y_clear)
        self.df_kd_final_sensor.loc[pathname, "CLEAR"] = slope * (-1)
        #print("kd" + str(self.df_kd_final_sensor.loc[pathname, "CLEAR"]))

        self.df_kd_final_sensor.to_csv("distances/" + file_name + "_0.3_0.6_0.9_1.25_double_sensor_error" + error +".csv")

        return self.df_kd_final_sensor.loc[pathname, "RED"], self.df_kd_final_sensor.loc[pathname, "GREEN"], \
               self.df_kd_final_sensor.loc[pathname, "BLUE"], self.df_kd_final_sensor.loc[pathname, "CLEAR"]

    def create_error(self, error):
        err = float(error)/2
        return random.uniform(-float(err), float(err))

    def calc_error(self, error, value_calc_error):
        return (value_calc_error*error)/100

    def create_RGB_wavelength(self):

        wlB, sensB1 = np.loadtxt('RGB_sensor_values/Blue_dataset.csv', dtype={'names': ('a', 'b'), 'formats': ('f4', 'f4')}, comments='#',
                                 delimiter=',', usecols=(0, 1), unpack=True)
        wlR, sensR1 = np.loadtxt('RGB_sensor_values/Red_dataset.csv', dtype={'names': ('a', 'b'), 'formats': ('f4', 'f4')}, comments='#',
                                 delimiter=',', usecols=(0, 1), unpack=True)
        wlG, sensG1 = np.loadtxt('RGB_sensor_values/Green_dataset.csv', dtype={'names': ('a', 'b'), 'formats': ('f4', 'f4')},
                                 comments='#', delimiter=',', usecols=(0, 1), unpack=True)
        wlC, sensC1 = np.loadtxt('RGB_sensor_values/Clear_dataset.csv', dtype={'names': ('a', 'b'), 'formats': ('f4', 'f4')},
                                 comments='#', delimiter=',', usecols=(0, 1), unpack=True)

        wl = np.arange(352.5, 795, 5)

        fB = interpolate.interp1d(wlB, sensB1, 'slinear')
        sensB = fB(wl)

        fR = interpolate.interp1d(wlR, sensR1, 'slinear')
        sensR = fR(wl)

        fG = interpolate.interp1d(wlG, sensG1, 'slinear')
        sensG = fG(wl)

        fC = interpolate.interp1d(wlC, sensC1, 'slinear')
        sensC = fC(wl)

        sensR_normalized = sensR / max(sensR)
        sensG_normalized = sensG / max(sensG)
        sensB_normalized = sensB / max(sensB)
        sensC_normalized = sensC / max(sensC)

        self.color_red_dictionary = dict(zip(wl, sensR_normalized))
        self.color_green_dictionary = dict(zip(wl, sensG_normalized))
        self.color_blue_dictionary = dict(zip(wl, sensB_normalized))
        self.clear_dictionary = dict(zip(wl, sensC_normalized))

    def cluster_all(self):
        root_path = os.getcwd()
        path = os.path.join(root_path, "distances")
        file_csv = self.find_file('all_wavelengths.csv', path)

        with open(file_csv) as file:

            data = file.read()
            data = data.replace("\\", ",").replace(",,", ",").replace(".txt", "").replace("\"", "").\
                replace("name", 'date,bottom,depth,chl,cdom,mineral,cloud,suntheta,windspeed,temp,salinity').\
                replace("bottom_", "").replace("depth_", "").replace("chl_", "").replace("cdom_", "").\
                replace("mineral_", "").replace("cloud_", "").replace("suntheta_", "").replace("windspeed_", "").\
                replace("temp_", "").replace("salinity_", "")

            with open("all_wavelengths.csv", "w") as outputfile:
                outputfile.write(data)
                outputfile.close()

        df = pd.read_csv("all_wavelengths.csv")
        df['name'] = range(1, len(df) + 1)
        df.to_csv("distances/all_wavelengths.csv", index=False)

        os.remove("all_wavelengths.csv")

        '''
        data = Orange.data.Table('distances/all_wavelengths.csv')
        print(data[0]["352.5"])
        for d in data[:3]:
            print(d)
        #print(data)
        '''

    def cluster_all_sensor(self):
        root_path = os.getcwd()
        path = os.path.join(root_path, "distances")
        file_csv = self.find_file('all_sensors.csv', path)

        with open(file_csv) as file:

            data = file.read()
            data = data.replace("\\", ",").replace(",,", ",").replace(".txt", "").replace("\"", "").\
                replace("name", 'date,bottom,depth,chl,cdom,mineral,cloud,suntheta,windspeed,temp,salinity').\
                replace("bottom_", "").replace("depth_", "").replace("chl_", "").replace("cdom_", "").\
                replace("mineral_", "").replace("cloud_", "").replace("suntheta_", "").replace("windspeed_", "").\
                replace("temp_", "").replace("salinity_", "").replace("RED,GREEN,BLUE", "BLUE,GREEN,RED")

            with open("all_sensors.csv", "w") as outputfile:
                outputfile.write(data)
                outputfile.close()

        df = pd.read_csv("all_sensors.csv")
        df['name'] = range(1, len(df) + 1)
        df.to_csv("distances/all_sensors.csv", index=False)

        os.remove("all_sensors.csv")

        '''
        data = Orange.data.Table('distances/all_sensors.csv')
        for d in data[:3]:
            print(d)
        #print(data)
        '''

    def cluster_distances_sensor(self):
        root_path = os.getcwd()
        path = os.path.join(root_path, "distances")
        file_csv = self.find_file('sensor.csv', path)

        with open(file_csv) as file:
            data = file.read()
            data = data.replace("\\", ",").replace(",,", ",").replace(".txt", "").replace("\"", "").\
                replace("name", 'date,bottom,depth,chl,cdom,mineral,cloud,suntheta,windspeed,temp,salinity').\
                replace("bottom_", "").replace("depth_", "").replace("chl_", "").replace("cdom_", "").\
                replace("mineral_", "").replace("cloud_", "").replace("suntheta_", "").replace("windspeed_", "").\
                replace("temp_", "").replace("salinity_", "").replace("RED,GREEN,BLUE", "BLUE,GREEN,RED")

            with open("distances_sensor.csv", "w") as outputfile:
                outputfile.write(data)
                outputfile.close()

        df = pd.read_csv("distances_sensor.csv")
        df['name'] = range(1, len(df) + 1)
        df.to_csv("distances/distances_sensor.csv", index=False)

        os.remove("distances_sensor.csv")

        '''
        data = Orange.data.Table('distances/distances_sensor.csv')
        for d in data[:3]:
            print(d)
        #print(data)
        '''

    def cluster_distances_sensor_error_20(self):
        root_path = os.getcwd()
        path = os.path.join(root_path, "distances")
        file_csv = ""
        filename = ""

        if self.find_file('sensor_error20.csv', path):
            filename = "distances_sensor_error20.csv"
            file_csv = self.find_file('sensor_error20.csv', path)

        with open(file_csv) as file:
            data = file.read()
            data = data.replace("\\", ",").replace(",,", ",").replace(".txt", "").replace("\"", "").\
                replace("name", 'date,bottom,depth,chl,cdom,mineral,cloud,suntheta,windspeed,temp,salinity').\
                replace("bottom_", "").replace("depth_", "").replace("chl_", "").replace("cdom_", "").\
                replace("mineral_", "").replace("cloud_", "").replace("suntheta_", "").replace("windspeed_", "").\
                replace("temp_", "").replace("salinity_", "").replace("RED,GREEN,BLUE", "BLUE,GREEN,RED")

            with open(filename, "w") as outputfile:
                outputfile.write(data)
                outputfile.close()

        df = pd.read_csv(filename)
        df['name'] = range(1, len(df) + 1)
        df.to_csv("distances/" + filename, index=False)

        os.remove(filename)
        '''    
        data = Orange.data.Table('distances/distances_sensor_error_20.csv')
        for d in data[:3]:
            print(d)
        #print(data)
        '''

    def cluster_distances_sensor_error_10(self):
        print("prova error 10")
        root_path = os.getcwd()
        path = os.path.join(root_path, "distances")
        file_csv = ""
        filename = ""

        if self.find_file('sensor_error10.csv', path):
            filename = "distances_sensor_error10.csv"
            file_csv = self.find_file('sensor_error10.csv', path)

        with open(file_csv) as file:
            data = file.read()
            data = data.replace("\\", ",").replace(",,", ",").replace(".txt", "").replace("\"", "").\
                replace("name", 'date,bottom,depth,chl,cdom,mineral,cloud,suntheta,windspeed,temp,salinity').\
                replace("bottom_", "").replace("depth_", "").replace("chl_", "").replace("cdom_", "").\
                replace("mineral_", "").replace("cloud_", "").replace("suntheta_", "").replace("windspeed_", "").\
                replace("temp_", "").replace("salinity_", "").replace("RED,GREEN,BLUE", "BLUE,GREEN,RED")

            with open(filename, "w") as outputfile:
                outputfile.write(data)
                outputfile.close()

        df = pd.read_csv(filename)
        df['name'] = range(1, len(df) + 1)
        df.to_csv("distances/" + filename, index=False)

        os.remove(filename)
        '''    
        data = Orange.data.Table('distances/distances_sensor_error_20.csv')
        for d in data[:3]:
            print(d)
        #print(data)
        '''

    def cluster_distances_double_sensor_error_20(self):
        print("prova double error 20")
        root_path = os.getcwd()
        path = os.path.join(root_path, "distances")
        file_csv = ""
        filename = ""

        if self.find_file('double_sensor_error20.csv', path):
            filename = "distances_double_sensor_error20.csv"
            file_csv = self.find_file('double_sensor_error20.csv', path)

        with open(file_csv) as file:
            data = file.read()
            data = data.replace("\\", ",").replace(",,", ",").replace(".txt", "").replace("\"", "").\
                replace("name", 'date,bottom,depth,chl,cdom,mineral,cloud,suntheta,windspeed,temp,salinity').\
                replace("bottom_", "").replace("depth_", "").replace("chl_", "").replace("cdom_", "").\
                replace("mineral_", "").replace("cloud_", "").replace("suntheta_", "").replace("windspeed_", "").\
                replace("temp_", "").replace("salinity_", "").replace("RED,GREEN,BLUE", "BLUE,GREEN,RED")

            with open(filename, "w") as outputfile:
                outputfile.write(data)
                outputfile.close()

        df = pd.read_csv(filename)
        df['name'] = range(1, len(df) + 1)
        df.to_csv("distances/" + filename, index=False)

        os.remove(filename)
        '''    
        data = Orange.data.Table('distances/distances_sensor_error_20.csv')
        for d in data[:3]:
            print(d)
        #print(data)
        '''

    def cluster_distances_double_sensor_error_10(self):
        print("prova double error 10")
        root_path = os.getcwd()
        path = os.path.join(root_path, "distances")
        file_csv = ""
        filename = ""

        if self.find_file('double_sensor_error10.csv', path):
            filename = "distances_double_sensor_error10.csv"
            file_csv = self.find_file('double_sensor_error10.csv', path)
            print(file_csv)

        with open(file_csv) as file:
            data = file.read()
            data = data.replace("\\", ",").replace(",,", ",").replace(".txt", "").replace("\"", "").\
                replace("name", 'date,bottom,depth,chl,cdom,mineral,cloud,suntheta,windspeed,temp,salinity').\
                replace("bottom_", "").replace("depth_", "").replace("chl_", "").replace("cdom_", "").\
                replace("mineral_", "").replace("cloud_", "").replace("suntheta_", "").replace("windspeed_", "").\
                replace("temp_", "").replace("salinity_", "").replace("RED,GREEN,BLUE", "BLUE,GREEN,RED")

            with open(filename, "w") as outputfile:
                outputfile.write(data)
                outputfile.close()

        df = pd.read_csv(filename)
        df['name'] = range(1, len(df) + 1)
        df.to_csv("distances/" + filename, index=False)

        os.remove(filename)
        '''    
        data = Orange.data.Table('distances/distances_sensor_error_20.csv')
        for d in data[:3]:
            print(d)
        #print(data)
        '''

    def find_file(self, pattern, path):
        for root, dirs, files in os.walk(path):
            for name in files:
                if re.search(pattern, name):
                    return os.path.join(root, name)



