import re


def find_data_txt(files):
    pattern = re.compile('^(?!.*converted).*\.txt$')
    return [file for file in files if pattern.match(file)]



if __name__ == '__main__':
    files = ['opensignals_0007800F2FB7_17-14-03_converted.txt', 
    'opensignals_0007800F2FB7_2023-09-25_17-14-03.edf',
    'opensignals_0007800F2FB7_2023-09-25_17-14-03.h5', 
    'opensignals_0007800F2FB7_2023-09-25_17-14-03.txt']

    print(find_data_txt(files))