import tensorflow as tf

def load_dataset_from_gdrive(FILENAME_dataset, compression_type):
    dataset = tf.data.TFRecordDataset(FILENAME_dataset, compression_type=compression_type)
    return dataset

# later
def download_dataset_from_waymo():
    pass

# Saving data to GDrive
def save_checkpoint_to_gdrive(model, drive):
    # Create & upload a text file.
    uploaded = drive.CreateFile({'title': 'Sample file.txt'})
    uploaded.SetContentString('Sample upload file content')
    uploaded.Upload()
    print('Uploaded file with ID {}'.format(uploaded.get('id')))

def reload_last_checkpoint():
    pass

# Get 4x3 Channel Lidar Data
# All: Reflectance and Elongation Values +
# 1.) HxW w/ Range values
# 2.) RxW w/ Height values
# 3.) RxH w/ Width: first value from right
# 4.) RxH w/ Width: first value from right
def get_two_point_five_dim_lidar_data(lidar_point_cloud):
    pass

# Get RGB Channel Data at corresponding lidar recording locations
# TODO: Register RGB and Lidar Data
def get_rgb_corresponding_to_lidar(lidar_data):
    pass

