import os


def combine_files(folder_path, file_out):
    files = os.listdir(folder_path)
    i = 0
    for file in files:
        if not os.path.isdir(file):
            with open(folder_path + '/' + file, 'r', encoding='utf-8')as file_pre, \
                    open(file_out, 'a', encoding='utf-8')as file_text:
                file_text.write(''.join(file_pre.readlines()) + '\n')
                i = i + 1
    print(folder_path, '的总样本数:', i)


# combine_files('E:/BJTUNLP_Practice2021/任务一/aclImdb/train/neg', 'E:/BJTUNLP_Practice2021/任务一/train_neg_text.txt')
# combine_files('E:/BJTUNLP_Practice2021/任务一/aclImdb/train/pos', 'E:/BJTUNLP_Practice2021/任务一/train_pos_text.txt')

combine_files('E:/BJTUNLP_Practice2021/任务一/aclImdb/test/neg', 'E:/BJTUNLP_Practice2021/任务一/test_neg_text.txt')
combine_files('E:/BJTUNLP_Practice2021/任务一/aclImdb/test/pos', 'E:/BJTUNLP_Practice2021/任务一/test_pos_text.txt')



