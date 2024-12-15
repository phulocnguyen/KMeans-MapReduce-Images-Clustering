from glob import glob
from os.path import join, isfile

def copy_contents_to_result(path):

    if not isfile(path):
        files = glob(join(path, 'part-r-*[0-9]'))
    else:
        files = [path]
    with open('/Users/phulocnguyen/Documents/Workspace/KMeans-MapReduce-Images-Clustering/resources/output/Clustering/result.txt', 'w', encoding='utf-8') as output:
        for file in files:
            try:
                with open(file, 'r', encoding='utf-8') as input_file:
                    content = input_file.read()
                    output.write(content)
                    output.write('\n')
                print(f'Đã copy nội dung từ file: {file}')
            except Exception as e:
                print(f'Lỗi khi đọc file {file}: {str(e)}')


path = '/Users/phulocnguyen/Documents/Workspace/KMeans-MapReduce-Images-Clustering/resources/output/Clustering/5'  
copy_contents_to_result(path)