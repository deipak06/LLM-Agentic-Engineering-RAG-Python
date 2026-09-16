# Take a sentence (hardcode it) and 
# reverse the words' order 
# (not the letters — "I like dogs" → "dogs like I")
words="I like dogs"
reverseword=""
splitword=words.split()

for word in splitword:
    if reverseword == "":
        reverseword=word
    else:
        reverseword=word+" "+reverseword 
print(reverseword)

# Write a function that checks if a string is a palindrome

def palindrome(word):
    return word==word[::-1]


    

print(palindrome("NITIN"))


# Read a .txt file (any file — even this roadmap file works)
#  and print the total number of lines and words

total_lines = 0
total_words = 0

with open("roadmap.txt", "r",encoding="utf-8") as file:
    for line in file:
        total_lines += 1
        total_words += len(line.split())

print(f"Total lines: {total_lines}")
print(f"Total words: {total_words}")
