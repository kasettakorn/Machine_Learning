import matplotlib.pyplot as plt
import copy
import statistics as stat
import math

unsorted_data = [21,24,8,9,15,21,26,28,25,34,29,4]
x_axis = [i for i in range(12)]
plt.plot(x_axis, unsorted_data, 'o--')


variance = stat.variance(unsorted_data, stat.mean(unsorted_data))
print("Variance = ", variance)


#sorting data
sorted_data = copy.deepcopy(unsorted_data)
sorted_data.sort()
print(stat.median(sorted_data))

#---binning method by median---

#divide data into 4 element
binning_data = []
binning_data.append(sorted_data[0:4])
binning_data.append(sorted_data[4:8])
binning_data.append(sorted_data[8:len(sorted_data)])
plot_by_median = copy.deepcopy(binning_data)
plot_by_boundaries = copy.deepcopy(binning_data)

#calculate median each list of list
for i in range(len(plot_by_median)):
    plot_by_median[i] = [math.ceil(stat.median(plot_by_median[i]))] * len(plot_by_median[i])

#calculate boundaries each list of list
for i in range(len(plot_by_boundaries)):
    for j in range(1, len(plot_by_boundaries[i])-1):
        if abs(plot_by_boundaries[i][i]-plot_by_boundaries[i][0]) < abs(plot_by_boundaries[i][i]-plot_by_boundaries[i][len(plot_by_boundaries[i])-1]):
            plot_by_boundaries[i][j] = plot_by_boundaries[i][0]
        else:
            plot_by_boundaries[i][j] = plot_by_boundaries[i][len(plot_by_boundaries[i])-1]

#Convert list of list into one list
plot_by_median  = [value for sublist in plot_by_median for value in sublist] 
plot_by_boundaries = [value for sublist in plot_by_boundaries for value in sublist]

#find closest of number in plot_by_median
temp = []
for data in unsorted_data:
    closet_value = 99999
    for mean_value in plot_by_median:
        if abs(mean_value-data) < closet_value:
            closet_value = abs(mean_value-data) 
            closet_data = mean_value 
    temp.append(closet_data)
    plot_by_median.remove(closet_data)
plot_by_median = temp
plt.plot(x_axis, plot_by_median, 'o-')
#end of binning by median

#find closest of number in plot_by_boundaries
temp = []
for data in unsorted_data:
    closet_value = 99999
    for mean_value in plot_by_boundaries:
        if abs(mean_value-data) < closet_value:
            closet_value = abs(mean_value-data) 
            closet_data = mean_value 
    temp.append(closet_data)
    plot_by_boundaries.remove(closet_data)
plot_by_boundaries = temp
plt.plot(x_axis, plot_by_boundaries, 'o-')

#---end of binning by boundaries---

#---information graph--

#plt.plot(x_axis, sorted_data, 'o-')
plt.legend(["Unsorted data", "Binning by median", "Binning by boundaries"])
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Binning data graph")
plt.text(8, 38, ('Median (x̃) = 22.5'))
plt.text(8, 36, 'Variance (σ²) = 87.6969')
plt.show()
