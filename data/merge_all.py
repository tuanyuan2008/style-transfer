# Python program for merging two files.
  
data = data2 = ''
  
# Reading data from file1 
with open('processed_files/dev/en_2.txt') as fp: 
    data = fp.read() 
  
# Reading data from file2 
with open('processed_files/dev/trump.txt') as fp: 
    data2 = fp.read() 
  
# Merging 2 files 
# To add the data of file2 
# from next line 
data += "\n"
data += data2 
  
with open ('processed_files/dev/all.txt', 'w') as fp: 
    fp.write(data) 

data = data2 = ''
  
# Reading data from file1 
with open('processed_files/test/en_2.txt') as fp: 
    data = fp.read() 
  
# Reading data from file2 
with open('processed_files/test/trump.txt') as fp: 
    data2 = fp.read() 
  
# Merging 2 files 
# To add the data of file2 
# from next line 
data += "\n"
data += data2 
  
with open ('processed_files/test/all.txt', 'w') as fp: 
    fp.write(data) 

data = data2 = ''
  
# Reading data from file1 
with open('processed_files/train/en_2.txt') as fp: 
    data = fp.read() 
  
# Reading data from file2 
with open('processed_files/train/trump.txt') as fp: 
    data2 = fp.read() 
  
# Merging 2 files 
# To add the data of file2 
# from next line 
data += "\n"
data += data2 
  
with open ('processed_files/train/all.txt', 'w') as fp: 
    fp.write(data) 