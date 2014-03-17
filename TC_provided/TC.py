#Elie Weintraub 
#ECE 467: Natural Language Processing
#
#TC.py - A program to train and test a text categorization system

import nltk, math, cPickle

###################################################################################################
#   Training Stage Definitions                                                                    #
###################################################################################################

class Token:
	""" A class that stores the IDF value and list of TF values for a given Token """
	def __init__(self):
		#Initialize data members 
		self.TF_dict = {}   # (key,value) => (category name, TF  of token for given category)
		self.doc_count = 0  # count of docs containing token used to compute the token's IDF
		self.IDF = 0        # token's IDF
		
	def setIDF(self,N):
		""" Computes and sets a token's IDF """
		self.IDF = math.log(float(N)/self.doc_count)
		
class InvertedIndex:
	""" A class that stores the inverted index generated when training the Text Classification System """
	def __init__(self):
		#Initialize data members 
		self.inverted_index = {} # (key,value) => (token, Token object)
		self.category_count = {} # (key,value) => (category, number of docs belonging to given category)
		self.N = 0               # total number of docs in training corpus
	
	def buildInvertedIndex(self): 
		""" Processes training files to create inverted index with unnormalized weights """
		#Prompt user for input file
		input_filename = raw_input('Please specify the file containing the list of labeled training documents: ')
		
		#Loop through docs, updating the inverted index (TF values & doc_counts of Token obj 
		#and N  and category_count of InvertedIndex obj)
		with open(input_filename,'r') as training_file_list:
			for line in training_file_list:  		
				file,category = line.split()
				self._updateInvertedIndex(file,category)
		
		#Set IDFs of each Token obj in the inverted index
			self._setIDFs()
	
	def normalizeWeights(self):
		""" Normalizes TF vectors of inverted index """
		normalization_constants = {} # dictionary to store the normalization constants of each category 
		
		#Loop through the inverted index, accumulating the sum-squared of TF*IDF weights for each category
		for token in self.inverted_index.keys():
			for category in self.inverted_index[token].TF_dict.keys():
				weight = self.inverted_index[token].TF_dict[category]*self.inverted_index[token].IDF
				normalization_constants[category] = normalization_constants.get(category,0) + weight**2
		
		#Take the square-root of the category sum-squared weights to get the normalization constants
		for category in normalization_constants.keys():
			normalization_constants[category] = math.sqrt(normalization_constants[category])
		
		#Loop through the inverted index, normalizing each TF by the appropriate normalization constant
		for token in self.inverted_index.keys():
			for category in self.inverted_index[token].TF_dict.keys():
				self.inverted_index[token].TF_dict[category]/=normalization_constants[category]
	
	def saveInvertedIndex(self):
		""" Saves an InvertedIndex object using the pickle module """
		#Prompt user for output filename
		output_filename = raw_input('Please specify the name for the file containing the trained system representation: ')
		#Save trained system representation 
		with open(output_filename,'w') as f:
			cPickle.dump(self,f)
		
	def _updateInvertedIndex(self,file,category):
		""" A helper function that updates the InvertedIndex obj based on the contents of the given file-label pair """
		token_list = self._getTokens(file)
		token_set = set(token_list)	
		#Iterate through token_list, incrementing TF of given category (label)
		for token in token_list:
			if token in self.inverted_index:
				self.inverted_index[token].TF_dict[category]=self.inverted_index[token].TF_dict.get(category,0)+1  
			else:
				self.inverted_index[token] = Token()
				self.inverted_index[token].TF_dict[category] = 1
		#Iterate through token_set, incrementing doc_count of given token
		for token in token_set:
			self.inverted_index[token].doc_count+=1
		#Increment N (total number of docs in corpus)
		self.N+=1
		#Update category_count
		self.category_count[category]=self.category_count.get(category,0)+1
	
	def _setIDFs(self):
		""" A helper function that sets the IDF of each token in the InvertedIndex object """
		for token in self.inverted_index.keys():
			self.inverted_index[token].setIDF(self.N)
	
	def _getTokens(self,filename):
		""" A helper function that tokenizes a  file and returns the list of tokens"""
		with open(filename,'r') as f:
			token_list = nltk.word_tokenize(f.read()) 			
		return token_list


#Driver Program
inverted_index=InvertedIndex()
inverted_index.buildInvertedIndex()
inverted_index.normalizeWeights()
inverted_index.saveInvertedIndex()
inverted_index2=loadInvertedIndex()	