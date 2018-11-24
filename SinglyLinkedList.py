import random
class node:
    def __init__(self, data=None):
        self.data = data
        self.next = None
class linked_list:
    def __init__(self):
        self.head = node()
    
    def append(self, data):
        newnode = node(data)
        currentNode = self.head
        while currentNode.next != None:
            currentNode = currentNode.next
        currentNode.next = newnode
    
    def size(self):
        currentNode = self.head
        length = 0
        while currentNode.next != None:
            length += 1
            currentNode = currentNode.next
        return length

    def print_linkedlist(self):
        elements = []
        currentNode = self.head
        while currentNode.next != None:
            currentNode = currentNode.next
            elements.append(currentNode.data)
        print (elements)

    def delete(self, data):
        currentNode = self.head
        while currentNode.next != None:
            tempNode = currentNode
            currentNode = currentNode.next
            if currentNode.data == data:
                tempNode.next = currentNode.next
                #currentNode.next = None
                return
            
        print(data, "is not found.")
           

mylist = linked_list() #create head node
for i in range(0,10):
    mylist.append(i)

mylist.print_linkedlist()
mylist.delete(7)
mylist.print_linkedlist()