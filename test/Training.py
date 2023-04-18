
# numbers={
#     "1":"One",
#     "2":"Two",
#     "3":"Three"
# }
# phone=input("Number:")
# size=len(phone)
# pos=0
# numFinal=[]
#
# while pos<size:
#     numFinal.append(numbers[phone[pos]])
#     pos+=1
# print(numFinal)

# message=input(">")
# words = message.split(" ")
# mojis = {
#     ":)":"😄",
#     ":(":"😌"
# }
# result=""
# for item in words:
#     result+=mojis.get(item,item)+" "
# print(result)

# def square(num):
#     return num*num
#
# try:
#
#     quad=square(int(input("Number:")))
#     print(quad)
#     print(200/quad)
# except ValueError:
#     print("Number conversion error")
# except ZeroDivisionError:
#     print("Division per zero error")

# class Person:
#     def __init__(self,name):
#         self.name=name
#     def talk(self):
#         print(self.name + " thank you")
#
# person=Person("Mario")
# person.talk()
import utils
# from utils import find_max
# try:
#     print(find_max(20))
# except Exception:
#     print("Error")

# from test_modules import dice
# import random
# print( f"({random.choice(dice.roll())} , {random.choice(dice.roll())})")

# from pathlib import Path
# path=Path()
# # for file in path.glob("*"):
# #     print(file)
#
# path.rmdir()

student=['Archana','krishna','Ramesh','vineeth']
def test(student1):
   # new={'alok':30,'Nevadan':28}
   # student.update(new)
   student1.append('Mario')
   print("Inside the function",student1)
   return
test(student)
print("outside the function:",student)