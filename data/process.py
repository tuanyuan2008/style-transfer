import re

with open('processed_files/dev/en.txt', 'r') as f, open('processed_files/dev/en_2.txt', 'w') as g:
    # line = re.sub(r'someword=|\,.*|\#.*','', f.read())
    # line = re.sub(r'\n+', '\n', line).strip()
    # g.writelines(line)
    for line in f:
        if line != '<POS> <CON_START>  <START>  <END>\n':
            g.write(line)
f.close()
g.close()

with open('processed_files/test/en.txt', 'r') as f, open('processed_files/test/en_2.txt', 'w') as g:
    # line = re.sub(r'someword=|\,.*|\#.*','', f.read())
    # line = re.sub(r'\n+', '\n', line).strip()
    # g.writelines(line)
    for line in f:
        if line != '<POS> <CON_START>  <START>  <END>\n':
            g.write(line)
f.close()
g.close()

with open('processed_files/train/en.txt', 'r') as f, open('processed_files/train/en_2.txt', 'w') as g:
    # line = re.sub(r'someword=|\,.*|\#.*','', f.read())
    # line = re.sub(r'\n+', '\n', line).strip()
    # g.writelines(line)
    for line in f:
        if line != '<POS> <CON_START>  <START>  <END>\n':
            g.write(line)
f.close()
g.close()