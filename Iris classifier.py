import matplotlib.pyplot as plt
import math

setosa_sepal = [3.0,3.2,3.1,3.6]
setosa_petal = [4.9,4.7,4.6,5.0]


versicolor_sepal = [3.2,3.1,2.3,2.8]
versicolor_petal = [6.4,6.9,5.5,6.5]

virginica_sepal = [3.0,2.6,3.0]
virginica_petal = [6.7,6.3,6.5]

#Plot graph
input_sepal = float(input("Enter sepal (cm): "))
input_petal = float(input("Enter petal (cm): "))
plt.subplot(1,2,1)
plt.title("Machine Learning แสดงการแยกสายพันธุ์ดอกไอริสด้วย 3-Nearest Neighbors\n(Iris classifier with 3-Nearest Neighbors)", 
            fontname='TH SarabunPSK', fontsize='18', x=1.08, fontweight="bold")
plt.plot(setosa_sepal, setosa_petal, 'bo')
plt.plot(versicolor_sepal, versicolor_petal, 'mo')
plt.plot(virginica_sepal, virginica_petal, 'go')
plt.plot(input_sepal, input_petal, 'r*')
plt.legend(["Iris-setosa", "Iris-versicolor", "Iris-virginica", "Input sample"])
plt.xlabel("ขนาดกลีบเลี้ยง (cm)", fontname='TH SarabunPSK', fontsize='14', fontweight="bold")
plt.ylabel("ขนาดกลีบดอก (cm)", fontname='TH SarabunPSK', fontsize='14', fontweight="bold")
plt.subplot(1,2,2)
plt.plot(setosa_sepal, setosa_petal, 'bo')
plt.plot(versicolor_sepal, versicolor_petal, 'mo')
plt.plot(virginica_sepal, virginica_petal, 'go')

#find 3-nearest neighbors
majority = []

for k in range(3):
    min_value = 99999
    for i in range(0, len(setosa_sepal)):
        euclidian = math.sqrt((setosa_sepal[i]-input_sepal)**2 + (setosa_petal[i]-input_petal)**2)
        if euclidian < min_value:
            min_value = euclidian
            min_type = "setosa"
            temp_sepal = setosa_sepal[i]
            temp_petal = setosa_petal[i]
            delete_index = i
          

    for i in range(0, len(versicolor_sepal)):
        euclidian = math.sqrt((versicolor_sepal[i]-input_sepal)**2 + (versicolor_petal[i]-input_petal)**2)
        if euclidian < min_value:
            min_value = euclidian
            min_type = "versicolor"
            temp_sepal = versicolor_sepal[i]
            temp_petal = versicolor_petal[i]   
            delete_index = i    

    for i in range(0, len(virginica_sepal)):
        euclidian = math.sqrt((virginica_sepal[i]-input_sepal)**2 + (virginica_petal[i]-input_petal)**2)
        if euclidian < min_value:
            min_value = euclidian
            min_type = "virginica"
            temp_sepal = virginica_sepal[i]
            temp_petal = virginica_petal[i]
            delete_index = i

    majority.append(min_type)
    if temp_sepal in setosa_sepal and temp_petal in setosa_petal:
        del setosa_sepal[delete_index]
        del setosa_petal[delete_index]
    elif temp_sepal in versicolor_sepal and temp_petal in versicolor_petal:
        del versicolor_sepal[delete_index]
        del versicolor_petal[delete_index]
    else:
        del virginica_sepal[delete_index]
        del virginica_petal[delete_index]

if majority.count("setosa") > majority.count("versicolor"):
    if majority.count("setosa") > majority.count("virginica"):
        plt.plot(input_sepal, input_petal, 'bo')
elif majority.count("versicolor") > majority.count("virginica"):
    plt.plot(input_sepal, input_petal, 'mo')
else:
    plt.plot(input_sepal, input_petal, 'go')



plt.legend(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
plt.xlabel("ขนาดกลีบเลี้ยง (cm)", fontname='TH SarabunPSK', fontsize='14', fontweight="bold")
plt.ylabel("ขนาดกลีบดอก (cm)", fontname='TH SarabunPSK', fontsize='14', fontweight="bold")
plt.show()