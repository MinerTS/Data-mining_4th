import math
import random

with open('/Users/lowhouse/Desktop/資料探勘/作業/第四次/glass.txt', 'r') as file:
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


def equalwidth(Class, class_index):
    
    interval_width = (max(Class) - min(Class)) / 10
    # 計算各個splitting points包含max, min
    Splitting_point = [min(Class) + i * interval_width for i in range(11)]
    Splitting_str = ["{:.4f}".format(num) for num in Splitting_point] # 調整浮點數顯示位數
        
    discretized_class = []

    # Perform equal width discretization
    for value in Class:
        for i in range(10):
            if Splitting_point[i] <= value < Splitting_point[i + 1]:
                discretized_class.append(i + 1)
            if value == max(Class):
                discretized_class.append(10)
                break
    class_name = column_names[Attributes[class_index]+1]  # 顯示類別名
    # print(f"Splitting Points(include Max. and Min.) for {class_name}:\n{Splitting_str}")

    return discretized_class


# print("\nRun Equal Width:\n")

discretized_row = []
discr_wid_data = []  #EquWid後的資料，List包含Dict裡面的value為str

# 循环处理每个属性
for class_selected in Attributes:
    valuesTodis = [float(row[class_selected]) for row in raw_data]
    discr_wid_values = equalwidth(valuesTodis, class_selected)  #執行EquWid

    discr_wid_str = []
    for value in discr_wid_values:
        str_value = str(value)
        discr_wid_str.append(str_value)

    discretized_row.append(discr_wid_str)

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

# for values in zip(*discretized_row):
#     discretized_dict = {key: value for key, value in zip(Attributes, values)}
#     discr_wid_data.append(discretized_dict)


#計算先驗機率(Train)

# attribute_count = [[[0 for _ in range(11)] for _ in range(9)] for _ in range(8)]
# class_count = [0 for _ in range(8)]

# for j in range(len(Ground_tru)):
#     for Ci in range(8):
#         if Ground_tru[j] == str(Ci):
#             class_count[Ci] += 1
#         for Ai in range(len(Attributes)):
#             for valueAi in range(1,11):
#                 if Ground_tru[j] == str(Ci) and int(discretized_row[Ai][j]) == valueAi:
#                     attribute_count[Ci][Ai][valueAi] += 1



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