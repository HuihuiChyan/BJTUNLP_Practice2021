import re
import pdb
import os
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

#data_processing

dataset_path = "."


def build_corpus(split, dont_test = True):
	text_path = split + '/' + split + '.txt'
	data_dir = dataset_path + '/' + text_path
	with open(data_dir, 'w', encoding='utf-8') as fgoal:
		if dont_test:
			for name in ['neg', 'pos']:
				file_name_list = os.listdir(dataset_path + "/" + split + "/" + name)
				for f_name in file_name_list:
					with open(os.path.join(dataset_path, split, name,f_name), "r", encoding='utf-8') as ftext:
						line = ftext.read()
						line = data_proc(line)
						fgoal.write(name + ' ' + line + '\n')
		else:
			with open(os.path.join(dataset_path, text_path), "r", encoding='utf-8') as ftext:
				lines = ftext.readlines()
				for line in lines:
					line = data_proc(line)
					fgoal.write(line + '\n')


def data_proc(text):
	#pdb.set_trace()
	#处理标点\缩略词\特殊字符
	text.replace("<br />", "")
	text = text.lower()
	text = text.replace("&", " and ").replace("'s", " is").replace("'m", " am")\
		.replace("'re", " are").replace("won't","will not").replace("can't", "can not")\
		.replace("n't", " not").replace("'ve"," have").replace("'ll"," will")
	text = re.sub(r"[^a-zA-Z0-9]+", " ", text)

	#停用词处理&词形还原/Lemmatisation
	stop_words = set(stopwords.words("english"))
	lem = WordNetLemmatizer()
	text = text.split()
	text = [lem.lemmatize(word, pos='v') for word in text]
	text = " ".join(text)

	return text

def merge_file(split_1, split_2):

	train_path = dataset_path + "/" + split_1 + "/" + split_1 + ".txt"
	dev_path = dataset_path + "/" + split_2 + "/" + split_2 + ".txt"

	outfile = 'result.txt'
	out_path = dataset_path+ "/" + outfile
	#files = os.listdir(path)
	print(train_path)
	print(dev_path)
	with open(out_path, "a+", encoding= 'utf-8') as fgoal,\
		open(train_path, 'r', encoding='utf-8') as ftrain,\
		open(dev_path, 'r', encoding='utf-8') as fdev:
		fgoal.write(ftrain.read())
		fgoal.write(fdev.read())
	#pdb.set_trace()
	print('合并完成')


if __name__ == '__main__':

    build_corpus("train")
    build_corpus("dev")

    merge_file("train", "dev")
    #build_corpus("test", dont_test = False)





