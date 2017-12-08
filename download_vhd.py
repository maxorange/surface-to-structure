import glob, os, tarfile, urllib2
import numpy as np
import dicom
import cv2

dir_female = 'https://mri.radiology.uiowa.edu/VHDicom/VHFCT1mm'
dir_male = 'https://mri.radiology.uiowa.edu/VHDicom/VHMCT1mm'

files_female = ['VHF-Ankle.tar.gz', 'VHF-Head.tar.gz', 'VHF-Hip.tar.gz', 'VHF-Knee.tar.gz', 'VHF-Pelvis.tar.gz', 'VHF-Shoulder.tar.gz']
files_male = ['VHMCT1mm_Head.tar.gz', 'VHMCT1mm_Hip.tar.gz', 'VHMCT1mm_Pelvis.tar.gz', 'VHMCT1mm_Shoulder.tar.gz']

os.makedirs('./dataset/vhd/train/input')
os.mkdir('./dataset/vhd/train/output')
os.makedirs('./dataset/vhd/test/input')
os.mkdir('./dataset/vhd/test/output')
os.mkdir('./dataset/tmp')

for filename in files_female:
    print 'Downloading: ', filename
    response = urllib2.urlopen(os.path.join(dir_female, filename))
    data = response.read()
    with open(os.path.join('./dataset', filename), 'w') as f:
        f.write(data)

    data = tarfile.open(os.path.join('./dataset', filename), 'r:gz')
    data.extractall('./dataset/tmp')
    data.close()

for filename in files_male:
    print 'Downloading: ', filename
    response = urllib2.urlopen(os.path.join(dir_male, filename))
    data = response.read()
    with open(os.path.join('./dataset', filename), 'w') as f:
        f.write(data)

    data = tarfile.open(os.path.join('./dataset', filename), 'r:gz')
    data.extractall('./dataset/tmp')
    data.close()

# data = []
# for filename in glob.glob('./dataset/tmp/*/vhf_*.dcm'):
#     image = dicom.read_file(filename, force=True)
#     data.append(image.pixel_array)
#
# data = np.array(data).astype(np.float32)
# data -= data.min()
# data /= data.max()
# data *= 65535
# data = data.astype(np.uint16)
#
# for i, l in enumerate(data):
#     filename = os.path.join(pathname, '{0:04d}.png'.format(i+1))
#     cv2.imwrite(filename, l)
