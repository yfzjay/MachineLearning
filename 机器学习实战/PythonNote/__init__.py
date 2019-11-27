import json
f=open("test1",encoding="UTF8")
file1=json.load(f)
Input=file1["Inputs"]
inputAdd=[]
inputValue=[]
for sinput in Input:
   inputAdd.append(sinput['Address'])
   inputValue.append(sinput['Value'])
print(inputAdd)
print(inputValue)