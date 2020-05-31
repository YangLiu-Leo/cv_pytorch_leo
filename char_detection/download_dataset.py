import pandas as pd 
import os
import requests
import zipfile
import shutil
links = pd.read_csv('mchar_data_list_0515.csv')
print(links)
dir_name = 'dataset'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
for i,link in enumerate(links['link']):
    file_name = links['file'][i]
    print(file_name, '\t', link)
    file_name = dir_name+'/'+file_name
    if not os.path.exists(file_name):
        response = requests.get(link, stream = True)
        with open(file_name ,'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

zip_list = ['mchar_train', 'mchar_test_a', 'mchar_val']

for little_zip in zip_list:
    if not os.path.exists(dir_name+'/'+little_zip):
        zip_file = zipfile.ZipFile(dir_name+'/'+little_zip+'.zip', 'r')
        zip_file.extractall(path=dir_name)


if os.path.exists(dir_name + '/' + '__MACOSX'):
    shutil.rmtree(dir_name + '/' + '__MACOSX')
