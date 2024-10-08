import cv2
from pyzbar.pyzbar import decode

def read_barcode(frame):
    barcodes = decode(frame)
    barcode_data_list = []
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        barcode_data = barcode.data.decode('utf-8')
        barcode_type = barcode.type
        text = f"{barcode_data} ({barcode_type})"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        barcode_data_list.append((barcode_data, barcode_type))
    return barcode_data_list

def read_barcode_from_webcam():
    cap = cv2.VideoCapture(0)
    barcode_data_str = ""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        barcode_data_list = read_barcode(frame)
        if barcode_data_list:
            barcode_data_str = ", ".join([f"{data} ({type})" for data, type in barcode_data_list])
        cv2.imshow('Barcode Scanner', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return barcode_data_str

if __name__ == "__main__":
    barcode_data_str = read_barcode_from_webcam()
    print("Detected Barcodes:", barcode_data_str)
