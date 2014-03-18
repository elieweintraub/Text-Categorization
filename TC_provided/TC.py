#Elie Weintraub 
#ECE 467: Natural Language Processing
#
#TC.py - A program to train and test a text categorization system

import nltk, math, cPickle

def getTokens(filename):
	""" A helper function that tokenizes a  file and returns the list of tokens """
	with open(filename,'r') as f:
		token_list = nltk.word_tokenize(f.read()) 			
	return token_list

def loadInvertedIndex():
	""" Loads in trained system representation  and returns it as a InvertedIndex object """
	input_filename = raw_input('Please specify the file containing the trained system representation: ')
	with open(input_filename,'r') as f:
		return cPickle.load(f)	
		
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
		
	###################################################################################################
	#   Training Stage Definitions                                                                    #
	###################################################################################################	
	
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
		#Generate list of tokens for the given document and set of unique tokens
		token_list = getTokens(file)
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
	
	###################################################################################################
	#   Testing Stage Definitions                                                                     #
	###################################################################################################
			
	def categorizeTexts(self): 
		""" Categorizes test files """
		#Prompt user for input filename and output filename
		input_filename = raw_input('Please specify the file containing the list of test documents: ')
		output_filename = raw_input('Please specify the name for the file containing the labeled test documents: ')
		#Loop through docs and categorize them
		with open(input_filename,'r') as test_file_list:
			with open(output_filename,'w') as outfile:
				for doc in test_file_list: 
					self._categorize(doc.strip(),outfile)
	
	def _categorize(self,doc,outfile):
		""" Helper function used  to categorize a single document and write the results to the outfile """
		#Generate list of tokens for the given document
		token_list = getTokens(doc)
		#Compute similarity metric for each of the categories
		similarities = {}
		for category in self.category_count.keys():
			similarities[category] = self._sim(token_list,category)
		#Pick the category with highest similarity and write results to output file
		label=max(similarities,key=similarities.get)
		outfile.write(doc+' '+label+'\n')
	
	def _sim(self,token_list,category):
		""" Helper function that computes the actual similarity metric """
		doc_TFs={}
		similarity=0
		#Compute TFs of document tokens
		for token in token_list:
			doc_TFs[token]=doc_TFs.get(token,0)+1
		#Compute similarity metric	
		for token in doc_TFs:
			if token in self.inverted_index and category in self.inverted_index[token].TF_dict:
				category_TF=self.inverted_index[token].TF_dict[category]
				doc_TF=doc_TFs[token]
				IDF=self.inverted_index[token].IDF
				similarity+=category_TF*doc_TF*(IDF**2)
		return similarity		
####################################################################################################
#  Driver Program                                                                                  #
####################################################################################################
if __name__ == "__main__":
	print 'Welcome to the Text Categorization Program:'
	print '-------------------------------------------'
	train = raw_input('Would you like to load in a previously trained system (0) or train the text categorization system with a new training set (1): ')
	if train:
		inverted_index=InvertedIndex()
		inverted_index.buildInvertedIndex()
		inverted_index.normalizeWeights()
	
	save=raw_input('Would you like to save a representation of the trained system for future use (0=>no,1=>yes): ')	
	if save:
		inverted_index.saveInvertedIndex()
	
	test = raw_input('Would you like to categorize a test set (0=>no,1=>yes): ')	
	if test:
		if not train:
			inverted_index=loadInvertedIndex()
		inverted_index.categorizeTexts()
	
	if not (train or test):
		print "This probably isn't the program you are looking for"
	
	print "Thank you for using the Text Categorization Program"	
		