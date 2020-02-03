from skimage import img_as_float
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.color import gray2rgb
from scipy import fftpack,ndimage,misc
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
import numpy as np



#read image section
image = img_as_float(rgb2gray(imread('/Users/hazard/Library/Preferences/PyCharmCE2019.3/scratches/Tester_/Tester24/2-12.png')))
image = misc.imresize(image,64.0 / image.shape[0])

#spectral Residual
fft = fftpack.fft2(image)

logAmplitude = np.log(np.abs(fft))


phase = np.angle(fft)

avgLogAmp = uniform_filter(logAmplitude,size=4, mode="nearest")


spectralResidual = logAmplitude - avgLogAmp

saliencyMap = np.abs(fftpack.ifft2(np.exp(spectralResidual + 1j*phase)))**2


#After Effect
saliencyMap = ndimage.gaussian_filter(saliencyMap, sigma=0.5)
plt.figure(figsize=(7.5, 7.5), frameon=False)
plt.axis("off")
plt.box(False)

plt.imshow(saliencyMap)
#plt.savefig('file',dpi=300)


plt.show()
