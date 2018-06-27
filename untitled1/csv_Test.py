import csv



# 打印列表里的元素
def print_element(list):
    # print("==")
    for i in dict_reader:
        print(i)

    # 分割线
def cut_off():
    print("_" * 100)

# 文件路径
file_path = "e:/t.csv"
# 打开文件
with open(file_path) as file:
    print("代码块1：")
    reader = csv.reader(file)

    cut_off()
    print("直接打印reader方法获得的对象：",reader)
    cut_off()
    print("迭代打印reader方法获得的对象：")

    for v in reader:
        print(v)
    cut_off()
    # print("代码块2：")
    # dict_reader = csv.DictReader(file)
    # print("直接打印DictReader方法获得的对象：" ,dict_reader)
    # cut_off()
    # print("迭代打印DictReader方法获得的对象：")
    # for i in dict_reader:
    #     print(i)
    #
    # cut_off()
    #
    # # list = [v for v in dict_reader]
    # # print_element(list)
    # # print(list)
    file.close()
