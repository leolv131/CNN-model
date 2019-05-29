from nn import *
from image_reader import *
from keras.optimizers import SGD, RMSprop, Adam
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument("--PATH", default='E:/keras/DATA', help="path of mat file") #mat文件的路径
parser.add_argument("--DIRECTORY", default='E:/keras/DATA/short_beam', help="path of img")#img文件的路径
parser.add_argument("--NB_EPOCH", type=int, default=20, help="# of epoch") #迭代步数
parser.add_argument("--BATCH_SIZE", type=int, default=128, help="BATCH_SIZE")
parser.add_argument("--VERBOSE", type=int, default=1, help="VERBOSE")
parser.add_argument("--VALIDATION_SPLIT", type=float, default=0.2, help="VALIDATION_SPLIT") #验证数据数据量
parser.add_argument("--PERCENT_TESTNUMBER", type=float, default=0.2, help="PERCENT_TESTNUMBER") #测试数据数据量
parser.add_argument('--WIDTH', type=int, default=32, help='# of WIDTH')  #改变后的图片的宽度
parser.add_argument('--HEIGHT', type=int, default=32, help='# of HEIGHT')  #改编后的图片的高度
parser.add_argument('--N_SAMPLE', type=int, default=10000, help='N_SAMPLE')  #样本数量
parser.add_argument('--NB_CLASSES', type=int, default=1, help='NB_CLASSES')  #输出层维数
parser.add_argument('--IMG_CHANNELS', type=int, default=3, help='IMG_CHANNELS')  #图片的通道数

args = parser.parse_args() #用来解析命令行参数

def main():
	OPTIMIZER = Adam()
	INPUT_SHAPE = (args.WIDTH, args.HEIGHT, args.IMG_CHANNELS)
	#导入输入和输出数据
	D = DataReader()
	y_data = D.MatReader(args.PATH,'y_data',args.N_SAMPLE)
	index = D.MatReader(args.PATH,'img_index',args.N_SAMPLE)
	x_data = D.ImgReader_index(index,args.DIRECTORY,args.WIDTH,args.HEIGHT)
	#划分训练数据和测试数据
	x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=args.PERCENT_TESTNUMBER)
	#神经网络训练
	N = NN()
	model = N.LeNet(input_shape=INPUT_SHAPE, classes=args.NB_CLASSES)
	model.compile(loss="mse", optimizer=OPTIMIZER)

	model.fit(x_train, y_train, 
			batch_size=args.BATCH_SIZE, epochs=args.NB_EPOCH, 
			verbose=args.VERBOSE, validation_split=args.VALIDATION_SPLIT)
	#对测试数据进行预测
	model.save('model.h5')
	Prediction = model.predict(x_test)
	#查看模型精度指标
	for i in range(2000):
		print([round(Prediction[i][0],3),round(y_test[i][0],3)])
	E = criterion()
	R_square = E.Rsquare(Prediction,y_test)
	Accuracy = E.Accuracy(Prediction,y_test)
	Rmae = E.Rmae(Prediction,y_test)
	Raae = E.Raae(Prediction,y_test)
	print('~~~~~~~~~~~~~~模型精度~~~~~~~~~~~~~~~~~')
	print('R方：',  round(R_square,3))
	print('精度：', round(Accuracy,3))
	print('RMAE：', round(Rmae,3))
	print('RAAE:',  round(Raae,3))

main()
	
