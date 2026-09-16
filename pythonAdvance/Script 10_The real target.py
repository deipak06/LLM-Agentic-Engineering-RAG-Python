# Word frequency counter. Read a .txt file, 
# count how often each word appears (case-insensitive, strip punctuation), 
# print sorted from most to least frequent.
import string
with open("roadmap.txt","r",encoding="utf-8") as files:
    dicttxt={}
    for line in files:
        words=line.split()
        for word in words:
            cleanword=word.lower().strip(string.punctuation)
            if cleanword: 
                if cleanword in dicttxt:
                    dicttxt[cleanword]+=1
                else:
                    dicttxt[cleanword]=1

sorted_words=sorted(dicttxt.items(),key=lambda item: item[1],reverse=True)
for word, count in sorted_words:
    print(f"word is {word} and number of time {count}")
