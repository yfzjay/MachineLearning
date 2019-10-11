import re
#str="abc.568 奥德赛科技'sdsd:s dasd"
str=(open("../file/Ch04/email/ham/{}.txt".format(1),encoding='UTF-8').read())
List=re.split(r'\W',str)
print(List)