'''
tensorflow เขียนเทียบเท่า assembly
high-level API
    1. keras <- ขี่บน platform tensorflow ของ MIT เป็น community
    2. pytorch ของ Facebook
    3. mxnet ของ microsoft
    4. theano ML ver1

MSE ไม่เหมาะกับการจำแนกข้อมูล
cross-entropy function

'''
from keras.models import Sequential #Directed Graph
from keras.layers import Dense, Activation #node
#dense คือเส้นที่เชื่อมไปยังทุกๆ node ใน layer ถัดไป (fully connected layer)

model = Sequential()
model.add(Dense(64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("softmax"))

#sgd คือ gradient descent
# define loss and optimization algorithm 
model.compile(loss='mean_squared_error', optimizer='sgd')
print(model.summary()) #param คือจำนวนเส้น รวม bias



