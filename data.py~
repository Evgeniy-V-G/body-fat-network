import csv
import numpy as np
import os
import random
from math import floor
from tempfile import TemporaryFile


class DataBody():
	###############
	def __init__(self, experiment_name = 'default', input_file = 'bodyfat.csv' ,
				 test_percent=0.2,height_unit = 'cm',weight_unit = 'kg'):
		if os.path.exists(os.path.abspath('log/input/' + experiment_name + 
										  '.npz')):
			self.load_file(experiment_name)
		else:
			self.experiment_name = experiment_name
			self.input_file = input_file
			self.test_percent = test_percent
			
			self.read_csv(input_file)
			
			self.height_unit = height_unit
			self.weight_unit = weight_unit
			self.data_unit = self.unit_transformation(height_unit, weight_unit)
										 
			self.data_train, self.data_test = self.divide_train_test(
												   test_percent)

			self.save_file(experiment_name)
		
			
	###############
	def read_csv(self, input_file):
		with open(input_file, newline='') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
			first_row = True
			self.data_raw = np.array([])
			for row in spamreader:
				if first_row:
					headers = row[0].split(',')
					first_row = False
				else: 
					self.data_raw = np.append(self.data_raw, row[0].split(','))
			
		feature_number = len(headers)
		row_number = int(self.data_raw.size/feature_number)
		self.data_raw = np.resize(self.data_raw,[row_number,feature_number])
		self.data_raw = self.data_raw.astype(np.float)
		self.headers_raw = headers
	
	###############	
	def unit_transformation(self, height_target, weight_target,
							height_original = 'inches', 
							weight_original = 'lbs'):
		'''
			Principle: Transform everything to inches and lbs to then transform
			to the target unit 
		'''
		# Initializing
		data_unit = self.data_raw
		
		# Height Transform		    
		if height_original == 'inches':
			if height_target == 'cm':
				multiplier = 2.54
				data_unit[:,4] = self.data_raw[:,4]*multiplier
				self.height_unit = height_target
			else:
				self.height_unit = height_original
				
		# Weight Transform	
		if weight_original == 'lbs':
			if weight_target == 'kg':
				multiplier = 0.453592
				data_unit[:,3] = self.data_raw[:,3]*multiplier
				self.weight_unit = weight_target
			else:
				self.weight_unit = weight_original
		
		return data_unit

	###############
	def divide_train_test(self, test_percent):
		data_size = self.data_unit.shape[0]
		
		test_size = floor(data_size*test_percent)
		train_size = data_size - test_size
		
		np.random.shuffle(self.data_unit)
		
		data_train = self.data_unit[0:train_size,:]
		data_test = self.data_unit[train_size:-1,:]
		
		return data_train, data_test
	
	###############	
	def save_file(self,experiment_name):
		outfile = os.path.abspath('log/input/' + experiment_name)
		
		try:
			np.savez(outfile,data_train=self.data_train,
					 data_test = self.data_test,
					 experiment_name = self.experiment_name, 
					 input_file = self.input_file,
					 height_unit = self.height_unit,
					 weight_unit = self.weight_unit)
			print('Saving:' + outfile + '.npz')
		except OSError:
			print("Saving Error")
		
	
	###############			 
	def load_file(self,experiment_name):
		experiment_data = np.load(os.path.abspath('log/input/' + experiment_name 
												  + '.npz'))
		self.data_train = experiment_data['data_train']
		self.data_test = experiment_data['data_test']
		self.experiment_name = experiment_data['experiment_name']
		self.input_file = experiment_data['input_file']
		self.height_unit = experiment_data['height_unit']
		self.weight_unit = experiment_data['weight_unit']
		
		print('Load Experiment: ' + str(self.experiment_name))
	
a = DataBody(experiment_name = 'defaultkkkkk')	
