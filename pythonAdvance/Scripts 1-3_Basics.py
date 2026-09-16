
# Print numbers 1-100, but "Fizz" for multiples of 3, 
# "Buzz" for multiples of 5, "FizzBuzz" for both 
# (classic, but actually do it yourself, 
# don't look up the answer)

for i in range(1,101):
    if i%3 == 0 and i%5 ==0:
         print("FizzBuzz")
    elif i%3==0:
         print("Fizz")
    elif i%5==0:
         print("Buzz")
    else: print (f"{i}")

# Ask the user for their name and age via input(), 
# print a sentence using both

Name=input("Enter the name")
age=int(input("Enter the age"))
print(f"User name is {Name} and age {age}")

# Write a function that takes a list of numbers and 
# returns the average

def avgnumber(number):
     avg=sum(number)/len(number)
     return avg

print(avgnumber([1, 2, 3, 4, 5, 6]))
print(avgnumber([10, 20, 30])) 