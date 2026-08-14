# PhoWhisper Translator

Ứng dụng web nhận dạng giọng nói tiếng Việt bằng
[PhoWhisper](https://github.com/VinAIResearch/PhoWhisper), dịch sang tiếng Anh
và phát giọng đọc bản dịch.

## Chức năng

- Nhập văn bản tiếng Việt và dịch sang tiếng Anh.
- Ghi âm trực tiếp trên trình duyệt trong tối đa 10 giây.
- Nhận dạng tiếng Việt bằng `vinai/PhoWhisper-small`.
- Phát giọng đọc tiếng Anh bằng Edge TTS.
- Giao diện responsive cho máy tính và điện thoại.
- Xóa file ghi âm tạm ngay sau khi xử lý.
- Giữ luồng ghi âm bằng dòng lệnh để phục vụ thử nghiệm.

## Yêu cầu

- Python 3.10 trở lên.
- [FFmpeg](https://ffmpeg.org/download.html) có trong `PATH` để đọc bản ghi âm
  từ trình duyệt.
- Kết nối Internet trong lần đầu tải model và khi sử dụng dịch/TTS.
- GPU hỗ trợ CUDA là tùy chọn; ứng dụng tự động dùng CPU nếu không có GPU.

## Cài đặt

```powershell
cd pho-whisper-translator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Chạy ứng dụng web

```powershell
python web_app.py
```

Mở `http://127.0.0.1:5000` trên trình duyệt và cấp quyền microphone khi được
yêu cầu. Lần nhận dạng đầu tiên có thể mất thêm thời gian vì ứng dụng cần tải
model `vinai/PhoWhisper-small`.

## Chạy chế độ dòng lệnh

```powershell
python app.py
```

Chọn microphone từ danh sách; ứng dụng sẽ ghi âm 10 giây, nhận dạng, dịch và
tạo file giọng đọc trong thư mục `audio_file/`.

## Cấu trúc

```text
pho-whisper-translator/
|-- app.py                 # Logic PhoWhisper, dịch, TTS và chế độ CLI
|-- web_app.py             # Flask server và API
|-- requirements.txt
|-- static/
|   |-- app.js
|   `-- styles.css
`-- templates/
    `-- index.html
```
