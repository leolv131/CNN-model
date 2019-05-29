import os
from PIL import Image
from keras.preprocessing.image import img_to_array
import numpy as np 
import scipy.io as scio

class DataReader:
	def __init__(self):
		self.path = ''      #mat文件的路径
		self.directory = ''     #img的路径		
		self.filename = ''    #每张img的文件名
		self.n_sample = int()      #样本的数量
		self.width = int()     #改变后的图片宽
		self.height = int()   #改变后的图片高
		self.index = []

	def MatReader(self, path, filename, n_sample):
		path_1 = path + '/' + filename + '.mat'
		#print(path_1)
		data1 = scio.loadmat(path_1)
		P = data1[filename][0: n_sample]
		return P
	
	def ImgReader_index(self, index, directory, width, height):
		data = []
		for i in index:
			#print(i[0])
			img = Image.open( directory + '/' + str(i[0]) + '.jpg' )
			resized_image = img.resize((width, height), Image.ANTIALIAS)
			arr = np.asarray(resized_image, dtype=np.float32)       # 数组维度(128, 192, 3)
			arr = img_to_array(resized_image)                          # 数组维度(128, 192, 3)
			arr /= 125
			#print(arr)
			data.append(arr)
		result = np.array(data)
		return result
	
	def ImgReader(self, directory,width, height):    #width和height是改变后的图片的宽和高
		data = []
		for imgname in os.listdir(directory):    # 参数是文件夹路径 directory 
			#print(imgname)  
			img = Image.open( directory + '/' + imgname )
			resized_image = img.resize((width, height), Image.ANTIALIAS)
			arr = np.asarray(resized_image, dtype=np.float64)       # 数组维度(128, 192, 3)
			arr = img_to_array(resized_image)                          # 数组维度(128, 192, 3)
			data.append(arr)
		result = np.array(data)
		return result

class criterion:
	def __init__(self):
		self.Prediction = []
		self.y_test = []	
		
	def Accuracy(self, Prediction, y_test):
		ACCURACY = np.mean(1 - np.true_divide(np.fabs(np.subtract(Prediction,y_test)),y_test))
		return ACCURACY
	
	def Rsquare(self, Prediction, y_test):
		R_square = 1 - sum(np.square((Prediction-y_test)))/sum(np.square((y_test-np.mean(y_test))))
		return R_square[0]
	
	def Rmae(self, Prediction, y_test):
		RMAE = max(np.fabs(y_test - Prediction))/np.std(y_test,ddof=1)
		return RMAE[0]
	
	def Raae(self, Prediction, y_test):
		RAAE = np.mean(np.fabs(np.subtract(Prediction,y_test)))/np.std(y_test,ddof=1)
		return RAAE
