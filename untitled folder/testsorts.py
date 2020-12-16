from sorts import insertionSort, mergeSort, merge
import random

def random_list(length, a, b):
    a = 0
    b = 10
    randList = []
    for i in range(length):
        randList.append(random.randint(a,b))
    return randList

def is_sorted(list):
    #create a sorted list
    sortedList = list[:]
    sortedList.sort()
    if list == sortedList:
        return True
    return False


def test_insertion_sort(a, b, c):
    for i in range(2):
        for j in range(c-1):
            if i == 0:
                smallList = random_list(a, 0, 10)
                sortedSmall = insertionSort(smallList)
                if is_sorted(sortedSmall) != True:
                    return "Failed with length" + len(sortedSmall)
                print( "success insertion length " + str(a))
            if i == 1:
                largeList = random_list(b, 0, 10)
                sortedLarge = insertionSort(largeList)
                if is_sorted(sortedLarge) != True:
                    return "Failed with length" + len(sortedLarge)
                print( "success insertion length " + str(b))


def test_merge_sort(a, b, c):
    for i in range(2):
        for j in range(c-1):
            if i == 0:
                smallList = random_list(a, 0, 10)
                sortedSmall = mergeSort(smallList)
                if is_sorted(sortedSmall) != True:
                    return "Failed with length" + len(sortedSmall)
                print( "success merge length " + str(a))
            if i == 1:
                largeList = random_list(b, 0, 10)
                sortedLarge = mergeSort(largeList)
                if is_sorted(sortedLarge) != True:
                    return "Failed with length" + str(b)
                print( "success merge length " + str(b))

#to test both tests on random numbers
def test():
    print("----testing----")
    for i in range(10):
        a = random.randint(0, 10)
        b = random.randint(90, 100)
        c = random.randint(0, 100)
        test_merge_sort(a, b, c)
        test_insertion_sort(a, b, c)
        print("--------")
    print("----DONE----")
test()
