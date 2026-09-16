
# Write a function that takes a list of words and returns a dict of {word: length}
def wordlengths(words):
    dictword={}
    for word in words:
        wordlength=len(word)
        dictword[word]=wordlength
    return dictword

print(wordlengths(["cat","elephant"]))




# Write a function that takes two lists and 
# returns a dict zipping them together (no using zip() — do it manually with a loop, 
# then redo it with zip() and compare)

def combine(keys,values):
    combinedict={}
    for i in range(len(keys)):
        combinedict[keys[i]]=values[i]
    return combinedict
print(combine(["a", "b","c"], [1, 2, 3]))

def combine_zip(keys,values):
    return dict(zip(keys,values))
print(combine(["a", "b"], [1, 2,3]))



# Write a function with a default argument 
# and a *args parameter — anything, just get the syntax under your fingers

def greet(greeting="hello",*names):
    if not names:
        print(f"{greeting}")
        return 
    for key in names:
        print(f"{greeting} {key}")

greet("Hey","Nithin","Bas")
greet()