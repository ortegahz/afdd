import matplotlib.pyplot as plt
import numpy as np

# Open the binary file for reading
file_path = '/home/manu/mnt/ST8000DM004-2U91/afdd/data_plus/v0/#5_Uac_desktest_series_ARC_F250M_L80M_50RLOAD_20250321.BIN'
with open(file_path, 'rb') as fid:
    # Read the data as uint16
    data = np.fromfile(fid, dtype=np.uint16)

# Plot the data
plt.plot(data)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Data Plot')
plt.show()
