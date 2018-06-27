import os

# file = "D:\winrar临时文件夹\test.txt"
print("路径分隔符：", os.path.sep)
print("_"*50)
print("根目录：", os.path.altsep)
print("_"*50)
print("当前目录：", os.path.curdir)
print("_"*50)
print("父目录：", os.path.altsep)
print("_"*50)
print("绝对路径：", os.path.abspath(os.curdir))
print("_"*50)
# print("链接路径：", os.path.join("/q/a/", "/c/b/"))
# print("_"*50)
# print("把文件分割为目录和文件两个部分，以列表返回：", os.path.split(os.path.abspath(os.curdir)))