import copy
from collections import defaultdict

from student import Student


# def find_max(list):
#     max=list[0]
#     for num in list:
#         if(num>max):
#             max=num
#     return max
# students1=[Student('Mario1', 20), Student('Antonio1', 40)]
# students=[[Student('Mario0', 10), Student('Antonio0', 20)], [Student('Mario1', 10), Student('Antonio1', 20)]]
# students[0].pop(0)
# for st in students:
#     for st1 in st:
#         print(st1.name)

# global variable
c = 1

def add():

    # use of global keyword
    global c

    # increment c by 2
    c = c + 2

    print(c)

add()
# for l in students:
#     l.name=l.name+'1'
# print(students)
# invert1=students[::-1]
# for e, e1 in zip(invert1, students1):
#     e.name = e.name +'**'+ str(e1.age)
# for e, e1 in zip(students, students1):
#     print(e.name, '--', e1.name)
#
# print('BOOL=', defaultdict(bool))
# st1=copy.deepcopy(students)
# st1[0].name='Jose'
# print('Students:')
# for i in range(0, len(students)):
#     print(students[i].name)
#
# print('St1:')
# for i in range(0, len(st1)):
#     print(st1[i].name)
