import re

with open('dev/en.txt', 'r') as f, open('dev/en_2.txt', 'w') as g:
    line = re.sub(r'someword=|\,.*|\#.*','', f.read())
    line = re.sub(r'\n+', '\n', line).strip()
    g.writelines(line)