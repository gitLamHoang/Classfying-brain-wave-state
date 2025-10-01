import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy as sp
import time

# Hàm để tạo bộ lọc bandpass cho dữ liệu
def filter(data):
    band = [0.5 / (0.5 * 512), 40 / (0.5 * 512)]  # Tạo một bộ lọc bandpass với tần số cắt 0.5 Hz đến 40 Hz
    b, a = sp.signal.butter(5, band, btype='band', analog=False, output='ba')  # Tạo hệ số bộ lọc b và a
    data = sp.signal.lfilter(b, a, data)  # Lọc dữ liệu sử dụng bộ lọc
    return data

# Hàm để trích xuất đặc trưng từ dữ liệu
def FeatureExtract(y):
    flm = 512  # Tần số lấy mẫu
    L = len(y)  # Độ dài của tín hiệu

    Y = np.fft.fft(y)  # Thực hiện biến đổi Fourier nhanh (FFT)
    Y[0] = 0  # Đặt thành phần DC bằng không
    P2 = np.abs(Y / L)  # Tính phổ hai phía
    P1 = P2[:L // 2 + 1]  # Tính phổ một phía
    P1[1:-1] = 2 * P1[1:-1]  # Điều chỉnh phổ FFT
    # plt.plot(P1) # P1 là mảng FFT
    # plt.show()

    # Tìm chỉ số của các giá trị tần số giữa 0.5 Hz và 4 Hz
    f1 = np.arange(len(P1)) * flm / len(P1)
    indices1 = np.where((f1 >= 0.5) & (f1 <= 4))[0]
    delta = np.sum(P1[indices1])

    # Tìm chỉ số của các giá trị tần số giữa 4 Hz và 8 Hz
    indices1 = np.where((f1 >= 4) & (f1 <= 8))[0]
    theta = np.sum(P1[indices1])

    # Tìm chỉ số của các giá trị tần số giữa 8 Hz và 13 Hz
    indices1 = np.where((f1 >= 8) & (f1 <= 13))[0]
    alpha = np.sum(P1[indices1])

    # Tìm chỉ số của các giá trị tần số giữa 13 Hz và 30 Hz
    indices1 = np.where((f1 >= 13) & (f1 <= 30))[0]
    beta = np.sum(P1[indices1])

    # Tính các tỷ lệ đặc trưng
    abr = alpha / beta
    tbr = theta / beta
    dbr = delta / beta
    tar = theta / alpha
    dar = delta / alpha
    dtabr = (delta + theta) / (alpha + beta)

    # Tạo từ điển chứa các đặc trưng
    dict = {"delta": delta,
            "theta": theta,
            "alpha": alpha,
            "beta": beta,
            "abr": abr,
            "tbr": tbr,
            "dbr": dbr,
            "tar": tar,
            "dar": dar,
            "dtabr": dtabr
            }
    return dict

# Khởi tạo biến
x = 0
y = []
fs = 512  # Tần số lấy mẫu
k = 15 * fs  # Độ dài của 1 cửa sổ trượt (sliding window)

print("START!")  # In ra thông báo bắt đầu

# Đường dẫn đến file dữ liệu cần xét
path = ("Lam_tinhtao.txt") #path data brainwave
file = open(path, "r")  # Mở file dữ liệu
filename = "test_fft.h5"  # Đường dẫn đến file model, chạy file SVM.py trước

# Đọc và xử lý dữ liệu
while x < (180 * fs):  # Lặp qua các giá trị dữ liệu
    if x % fs == 0:
        print(x // fs)  # In ra số giây hiện tại
    data = file.readline()  # Đọc một dòng dữ liệu từ file
    y = np.append(y, int(data))  # Thêm dữ liệu vào mảng y
    x += 1  # Tăng giá trị x
    if x >= k:  # Khi đã thu thập đủ dữ liệu cho một cửa sổ trượt
        if x % (1 * 512) == 0:
            sliding_window_start = x - k  # Vị trí bắt đầu của cửa sổ trượt
            sliding_window_end = x  # Vị trí kết thúc của cửa sổ trượt
            sliding_window = np.array(y[sliding_window_start:sliding_window_end])  # Tạo mảng sliding_window tương ứng với cửa sổ trượt trong y
            # sliding_window = filter(sliding_window)  # Áp dụng bộ lọc (nếu cần)
            plt.plot(sliding_window)  # In biểu đồ dữ liệu raw
            plt.show()
            model = pickle.load(open(filename, 'rb'))  # Tải mô hình đã huấn luyện
            feature_test = np.array(list(FeatureExtract(sliding_window).values())).reshape(1, -1)  # Trích xuất đặc trưng và định hình lại để đưa vào mô hình
            print(feature_test)
            if model.predict(feature_test)[0] == 1:  # Dự đoán tình trạng tỉnh táo
                print('Tình trạng: Tỉnh táo ', model.predict(feature_test))
            else:  # Dự đoán tình trạng buồn ngủ
                print('Tình trạng: Buồn ngủ', model.predict(feature_test))

            time.sleep(1)  # Tạm dừng 1 giây

# Đóng file dữ liệu
file.close()
print("DONE")  # In ra thông báo hoàn thành