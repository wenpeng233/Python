def is_odd(num):
    return num % 2 == 1



if  __name__ ==  "__main__":
    list = []

    for i in range(1,51):   #range（a,b）等价于[a,b)区间的整数
        list.append(i)      #向列表添加元素
        print(i ,end="  ")  #end=""实现不换行

    print()                #换行

    print("*"* 100)

    for i in filter(is_odd,list):
        print(i,end="  ")
    print()
