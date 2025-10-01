from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Đọc dữ liệu từ các tệp CSV
file1 = pd.read_csv('testmo.csv', encoding='utf-8')  # Dữ liệu mắt mở
file2 = pd.read_csv('testnham.csv', encoding='utf-8')  # Dữ liệu mắt nhắm

# Kết hợp dữ liệu từ hai tệp
X = pd.concat([file1, file2])
# Tạo nhãn tương ứng (1: mở mắt, 0: nhắm mắt)
y = pd.concat([pd.Series([1] * len(file1)), pd.Series([0] * len(file2))])

# In ra dữ liệu đầu vào để kiểm tra
print(X)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Khởi tạo mô hình SVM với kernel tuyến tính
model = svm.SVC(kernel="linear")
# Huấn luyện mô hình với tập dữ liệu huấn luyện
model.fit(X_train.values, y_train.values)

# Lưu mô hình đã huấn luyện vào file
filename = 'test_fft.h5'
pickle.dump(model, open(filename, 'wb'))

# Tính điểm (score) của mô hình trên tập huấn luyện và tập kiểm tra
train_score = model.score(X_train, y_train)
score = model.score(X_test, y_test)
print("Train score: ", train_score)
print("Test score: ", score)

# Dự đoán nhãn trên tập dữ liệu kiểm tra
y_pred = np.array(model.predict(X_test))

# Tạo ma trận nhầm lẫn để đánh giá kết quả mô hình
label_mapping = {"Awake": 1, "Sleepy": 0}
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Tạo báo cáo phân loại chi tiết
clr = classification_report(y_test, y_pred, target_names=label_mapping.keys())

# Vẽ biểu đồ ma trận nhầm lẫn
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')
plt.xticks(np.arange(2) + 0.5, label_mapping.keys())
plt.yticks(np.arange(2) + 0.5, label_mapping.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()  # Hiển thị biểu đồ

# In báo cáo phân loại
print(f"Classification Report for {type(model).__name__}:\n----------------------\n", clr)

# # KFold Cross-Validation (mã mẫu, có thể kích hoạt nếu cần)
# from sklearn.model_selection import KFold
# from sklearn.metrics import accuracy_score
#
# # Định nghĩa số lượng fold
# k = 5
#
# # Khởi tạo đối tượng KFold
# kf = KFold(n_splits=k)
#
# # Khởi tạo danh sách để lưu điểm accuracy
# accuracy_list = []
#
# # Vòng lặp qua từng fold
# for train_index, test_index in kf.split(X):
#     # Chia dữ liệu thành tập huấn luyện và kiểm tra
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#
#     # Khởi tạo mô hình
#     model = svm.SVC(kernel="linear")
#
#     # Huấn luyện mô hình trên tập dữ liệu huấn luyện
#     model.fit(X_train, y_train)
#
#     # Dự đoán trên tập dữ liệu kiểm tra
#     y_pred = model.predict(X_test)
#
#     # Tính điểm accuracy
#     accuracy = accuracy_score(y_test, y_pred)
#
#     # Thêm điểm accuracy vào danh sách
#     accuracy_list.append(accuracy)
#
# # Tính điểm accuracy trung bình
# avg_accuracy = np.mean(accuracy_list)
# print("Average Accuracy Score:", avg_accuracy)