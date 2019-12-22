ĐÔ án tốt nghiệp: Mô hình nhận dạng cảm xúc tiếng nói Tiếng Việt

Các bước:
1. Trích xuất đặc trưng:
bash extract_features.sh
2. Huấn luyện mô hình:
bash train.sh
3. Đánh giá mô hình:
- với tập dữ liệu mới: chạy file extract_features.sh để trích xuất đặc trưng
- Với file wav bất kì: bash test_wav.sh
Note: 
- Các tham số để chạy file .sh đều được hướng dẫn chi tiết khi chạy.
- Để thay đổi các tham số đặc trưng: set up trong file scr/config.py
- Để cắt file wav thành các segment có overlap sử dụng file cut.sh




