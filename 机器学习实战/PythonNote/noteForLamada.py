from functools import reduce
##函数##
add=lambda x,y:x*y
print(add(2,3))
print((lambda x,y:x*y)(2,3))
##函数式编程##
##1.sorted
list1=[2,1,-3,0,-5]
list2=sorted(list1,key=lambda x:abs(x),reverse=True)
print(list2)
##2.map
list3=list(map(lambda x:x**3,list1))
print(list3)
##3.filter
list4=list(filter(lambda x:x%2!=0,range(1,21)))
print(list4)
##4.reduce
list5=reduce(lambda x,y:x+y,range(1,101))
print(list5)
##5.闭包
def sub1(n):
    return lambda x:x-n
sub2=sub1(5)##sub2代表n已经设置为5，只需传入x
print(sub2(15))


