import math
import random

with open('作業/第三次/glass.txt', 'r') as file:
    lines = file.readlines()
# 自定義column name
column_names = ['Id number', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe','Type of glass']
Attributes = [0,1,2,3,4,5,6,7,8]
# 初始化字典
raw_data = []
Ground_tru = []

for line in lines:
    row = line.strip().split(',')
    Ground_tru.append(row.pop(10))
    if len(row) == len(Attributes)+1:
            instance = dict(zip(Attributes, row[1:]))
            raw_data.append(instance)

def equalfrequency(Class, class_index):
    # 設定每個interval的instances#
    # interval_freq = len(Class) // 10
    # bin_sizes = [interval_freq] * 10
    # Sorted_class = sorted(Class)

    # # 於平均分配前四個
    # remainder = len(Class) % 10
    # for i in range(remainder):
    #     bin_sizes[i] += 1

    # # 計算SplittingPoint
    # Splitting_point = [0]
    # for size in bin_sizes:
    #     next_point = Splitting_point[-1] + size
    #     Splitting_point.append(next_point)

    # Splitting_intervals = [0] #紀錄intervals

    # discretized_class = Class
    # points_position = {}
    
    # for i in range(len(Class)):
    #     points_position[i] = Class[i]
    # Sorted_class = sorted(points_position.items(), key=lambda x: x[1])
    # for point in Splitting_point[1:10]:
    #     Splitting_value = (Sorted_class[point - 1][1] + Sorted_class[point][1]) / 2
    #     Splitting_intervals.append(Splitting_value)
    # Splitting_intervals.append(max(Class))

    # intervals_num = []
    # for i in range(0,10):
    #     intervals = []
    #     for j in range(Splitting_point[i], Splitting_point[i+1]):
    #         intervals.append(Sorted_class[j][0])
    #     intervals_num.append(intervals)
    
    # for i in range(10):
    #     label = intervals_num[i]
    #     for j in label:
    #         discretized_class[j] = i + 1

    #  設定每個interval的instances#
    interval_freq = len(Class) // 10
    bin_sizes = [interval_freq] * 10
    Sorted_class = sorted(Class)

    # 於平均分配前四個
    remainder = len(Class) % 10
    for i in range(remainder):
        bin_sizes[i] += 1

    # 計算SplittingPoint
    Splitting_point = [0]
    for size in bin_sizes:
        next_point = Splitting_point[-1] + size
        Splitting_point.append(next_point)

    Splitting_intervals = [0] #紀錄intervals
    for i in Splitting_point[1:10]:
        interval = (Sorted_class[i] + Sorted_class[i + 1]) / 2
        Splitting_intervals.append(interval)
    Splitting_intervals.append(Sorted_class[213])

    discretized_class = []
    for value in Class:
        for j in range(10):
            if Splitting_point[j] <= Sorted_class.index(value) < Splitting_point[j+1]:
                 discretized_class.append(j + 1)
            if value == max(Class):
                discretized_class.append(10)
                break

    # Splitting_str = ["{:.4f}".format(num) for num in Splitting_intervals] # 調整浮點數顯示位數
    # class_name = column_names[Attributes[class_index] + 1]  # match column name
    # print(f"Splitting Points(include Max. and Min.) for {class_name}:\n{Splitting_str}")

    return discretized_class
    

# print("\nRun Equal Frequency:\n")

discretized_row = []
discr_freq_data = []  #EquFreq後的資料，List包含Dict裡面的value為str

# 循环处理每个属性
for class_selected in Attributes:  # 跳过第一个和最后一个列名
    valuesTodis = [float(row[class_selected]) for row in raw_data]
    discr_freq_values = equalfrequency(valuesTodis, class_selected)  #執行EquFreq

    discr_freq_str = []
    for value in discr_freq_values:
        str_value = str(value)
        discr_freq_str.append(str_value)

    discretized_row.append(discr_freq_str)

for values in zip(*discretized_row):
    discretized_dict = {key: value for key, value in zip(Attributes, values)}
    discr_freq_data.append(discretized_dict)

# discr_row_train = discretized_row
# discr_row_train.append(Ground_tru)

# #計算先驗機率
# attribute_count = [[[0 for _ in range(11)] for _ in range(9)] for _ in range(8)]
# class_count = [0 for _ in range(8)]

# for j in range(len(Ground_tru)):
#     for Ci in range(8):
#         if Ground_tru[j] == str(Ci):
#                     class_count[Ci] += 1
#         for Ai in range(len(Attributes)):
#             for valueAi in range(1,11):
#                 if Ground_tru[j] == str(Ci) and int(discretized_row[Ai][j]) == valueAi:
#                     attribute_count[Ci][Ai][valueAi] += 1

# for j in range(214):
#     for c in range(8):
#                 if Ground_tru[j] == str(c):
#                     class_count[c] += 1

def laplace(target_class,attr,value):  
    laplace = (attribute_count[target_class][attr][value]+1) / (class_count[target_class] + 10)
    return laplace

#Naive Bayes classifier
def classifier(candidate_attr_set):
    Acc_count = 0

    for instance in range(len(run_testing_set[0])):
        predic = 0        
        best_likehood = 0

        for target_class in range(1,8):
            likehood = class_count[target_class] / (len(run_training_set[0]))

            for attr in candidate_attr_set:
                value = int(run_testing_set[attr][instance])
                likehood *= laplace(target_class, attr, value)

            if likehood > best_likehood:
                best_likehood = likehood
                predic = target_class

        if run_testing_set[9][instance] == str(predic):
            Acc_count += 1

    return Acc_count #/ len(Ground_tru)

# 5-fold cross validation
row_array = discretized_row
row_array.append(Ground_tru)
reverse_arry = []
for i in range(len(Ground_tru)):
    length = []
    for j in range(len(row_array)):
        length.append(row_array[j][i])
    reverse_arry.append(length)
post_row = reverse_arry
random.shuffle(post_row)

#  設定每個fold的instances#
fold_freq = len(Ground_tru) // 5
fold_sizes = [fold_freq] * 5

# 於平均分配前四個
remainder = len(Ground_tru) % 5
for i in range(remainder):
    fold_sizes[i] += 1

fold_point = [0]
for size in fold_sizes:
    next_point = fold_point[-1] + size
    fold_point.append(next_point)

# 開始分割
folds_data = []
for i in range(5):
    one_fold = []
    for j in range(len(Ground_tru)):
        if fold_point[i] <= j < fold_point[i + 1]:
            one_fold.append(post_row[j])
    folds_data.append(one_fold)

# 5-folds迴圈，分出training/testing set
data_set = folds_data
for i in range(len(folds_data)):
    testing_set = []
    training_set = []
    testing_set.extend(data_set[i])
    for j in range(len(folds_data)):
        if j != i:
            training_set.extend(data_set[j])
    
    run_testing_set = [] # 整理testing_set
    for i in range(len(testing_set[0])):
        app_testing_data = []
        for j in range(len(testing_set)):
            app_testing_data.append(testing_set[j][i])
        run_testing_set.append(app_testing_data)

    run_training_set = [] # 整理training_set
    for i in range(len(training_set[0])):
        app_training_data = []
        for j in range(len(training_set)):
            app_training_data.append(training_set[j][i])
        run_training_set.append(app_training_data)

    # 計算先驗機率(Train)
    attribute_count = [[[0 for _ in range(11)] for _ in range(9)] for _ in range(8)]
    class_count = [0 for _ in range(8)]

    for j in range(len(run_training_set[1])):
        for Ci in range(8):
            if run_training_set[9][j] == str(Ci):
                class_count[Ci] += 1
            for Ai in range(len(Attributes)):
                for valueAi in range(1,11):
                    if run_training_set[9][j] == str(Ci) and int(run_training_set[Ai][j]) == valueAi:
                        attribute_count[Ci][Ai][valueAi] += 1
    
    # classification
    acc_count = classifier(Attributes)

    print(f"Correct prediction: {acc_count}, Accuracy: {acc_count / len(run_testing_set[0]):.4f}")

# picked_attr = []
# rest_attr = []
# best_acc = 0
# new_best_feature = None

# for i in range(len(Attributes)):
#     rest_attr = set(Attributes) - set(picked_attr)
       
#     for attr in rest_attr:
#         candidate_attr = picked_attr + [attr]
#         candidate_attr_set = []

#         for features in candidate_attr:
#             # index = attribute_glass.index(features)
#             candidate_attr_set.append(features)

#         accuracy = classifier(candidate_attr_set)
        
#         if accuracy > best_acc:
#             best_acc = accuracy
#             new_best_feature = attr
             
#     if new_best_feature not in picked_attr:
#         picked_attr.append(new_best_feature)
#         print(f"Attrbutes subset: {[column_names[i + 1] for i in picked_attr]},Accuracy: {best_acc:.4f}")