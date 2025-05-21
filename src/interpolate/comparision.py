import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from wav import read_wav_data, write_wav_data
from cubic_spline import interpolate_sound
import time

print("Reading data...")
sample_rate, segment = read_wav_data('data/s5.wav')  
segment = np.array(segment, dtype=np.int16)
print(f"Sample rate: {sample_rate}, Segment length: {len(segment)} samples")

# Bước 3: Giả sử dữ liệu thiếu
print("Simulating missing data...")
np.random.seed(42)
x_known = np.sort(np.random.choice(len(segment), size=int(0.5*len(segment)), replace=False))
x_unknown = np.setdiff1d(np.arange(len(segment)), x_known)
# Lưu file bị nhiễu
faulty_segment = segment.copy()
faulty_segment[x_unknown] = 0  # Giả sử dữ liệu bị thiếu là 0
# Lưu file bị nhiễu
write_wav_data('data/faulty_segment.wav', faulty_segment, sample_rate)

print("Starting interpolation...")
start_time = time.time()
y_full = interpolate_sound(faulty_segment, x_known, samples_per_segment=sample_rate // 10)
print(f"Time taken for interpolation: {time.time() - start_time:.2f} seconds")

mse = np.mean((segment - y_full)**2)
print(f"MSE: {mse}")

# Bước 7: Lưu dữ liệu đã nội suy
output_file = 'data/interpolated.wav'
wavfile.write(output_file, sample_rate, y_full.astype(np.int16))

# Bước 6: Vẽ đồ thị
x_full = np.arange(len(segment))
y_known = segment[x_known]
plt.figure(figsize=(12, 6))
plt.plot(segment, label='Original Signal', alpha=0.75, color='red')
plt.plot(x_full, y_full, label='Interpolated Signal', linestyle='--', color='green')
plt.title('Cubic Spline Interpolation')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.show()


