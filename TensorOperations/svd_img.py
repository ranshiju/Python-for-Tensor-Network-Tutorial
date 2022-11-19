import numpy as np
import cv2
import matplotlib.pyplot as plt


M = cv2.imread('../data/images/img1.jpg')
print('type(x) = ', type(M))
print('X.shape = ', M.shape)
print('For the greyscale images, the data in the three '
      'channels are identical:')
print('Part of channel-0: \n', M[5:10, 5:10, 0])
print('Part of channel-1: \n', M[5:10, 5:10, 1])
print('Part of channel-2: \n', M[5:10, 5:10, 2])

print('--------------------- 分割线 ---------------------')

M = M[:, :, 0]
u, s, v = np.linalg.svd(M)
X1 = u[:, :1].dot(np.diag(s[:1])).dot(v[:1, :])
X2 = u[:, :10].dot(np.diag(s[:10])).dot(v[:10, :])
X3 = u[:, :20].dot(np.diag(s[:20])).dot(v[:20, :])
X4 = u[:, :50].dot(np.diag(s[:50])).dot(v[:50, :])
X5 = u[:, :100].dot(np.diag(s[:100])).dot(v[:100, :])

fig = plt.figure(1)
fig.set_size_inches(9, 10)  # set figure size
plt.subplot(321), plt.axis('off')
plt.title(r'$\tilde{r} = 1$')
plt.imshow(X1, cmap='gray')
plt.subplot(322), plt.axis('off')
plt.title(r'$\tilde{r} = 10$')
plt.imshow(X2, cmap='gray')
plt.subplot(323), plt.axis('off')
plt.title(r'$\tilde{r} = 20$')
plt.imshow(X3, cmap='gray')
plt.subplot(324), plt.axis('off')
plt.title(r'$\tilde{r} = 50$')
plt.imshow(X4, cmap='gray')
plt.subplot(325), plt.axis('off')
plt.title(r'$\tilde{r} = 100$')
plt.imshow(X5, cmap='gray')
plt.subplot(326), plt.axis('off')
plt.title('original')
plt.imshow(M, cmap='gray')
plt.savefig('img_svd.png', bbox_inches='tight', dpi=500)
plt.show()


